# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC. and
# Stanford University.
# All rights reserved.
# LLNL-CODE-826307

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from dataclasses import dataclass

import numpy as np

from .donut_model import DZMultiDonutModel, DZBasisMultiDonutModel
from .spot_model import DZMultiSpotModel, DZBasisMultiSpotModel


@dataclass
class ModelGroup:
    """A single sub-model with its weight and optional label.

    Parameters
    ----------
    model : BaseMultiDonutModel or BaseMultiSpotModel subclass
        Pre-constructed sub-model.
    weight : float, optional
        Relative weight for chi residuals.  The chi values are multiplied
        by ``sqrt(weight)`` so the cost contribution scales by ``weight``.
        Default 1.0.
    label : str or None, optional
        Human-readable label (e.g. ``"intra"``, ``"extra"``, ``"spots"``).
        Used only for debugging / bookkeeping.  Default None.
    """
    model: object
    weight: float = 1.0
    label: str | None = None

    @property
    def sqrt_weight(self):
        return np.sqrt(self.weight)


class MultiGroupJointModel:
    """Joint model with arbitrary groups sharing a single wavefront
    parameterization, each outer group having its own atmospheric kernel.

    Parameters
    ----------
    atm_groups : list of list of ModelGroup
        Outer list: each element is an *atm group* — the sub-models in
        that inner list share one atmospheric kernel block in the parameter
        vector.  Inner list: :class:`ModelGroup` instances within that atm
        group, each carrying its own observation weight.

    Notes
    -----
    Within each atm group all sub-models must have the same ``atm_mode``;
    across groups ``atm_mode`` may differ.  All sub-models must share the
    same ``wavefront_step``.

    Parameter layout (flat tuple)::

        [per_star_m0(3*n0) | per_star_m1(3*n1) | ...   (flat model order)
         atm_g0(natm_g0)   | atm_g1(natm_g1)   | ...   (one block / atm group)
         wavefront(nwf)                                  (shared)
         bkgs_m0           | bkgs_m1           | ...]   (flat model order)

    where ``per_star_m = [fluxes_m | dxs_m | dys_m]``.
    """

    def __init__(self, atm_groups):
        if not atm_groups:
            raise ValueError("atm_groups must not be empty")
        for g, ag in enumerate(atm_groups):
            if not ag:
                raise ValueError(f"atm group {g} is empty")

        self._atm_groups = atm_groups
        self._flat_groups = [mg for ag in atm_groups for mg in ag]

        # Within each atm group, all models must share the same atm_mode.
        for g, ag in enumerate(atm_groups):
            mode0 = ag[0].model.atm_mode
            for mg in ag[1:]:
                if mg.model.atm_mode != mode0:
                    raise ValueError(
                        f"atm_mode mismatch within atm group {g}: "
                        f"'{mode0}' vs '{mg.model.atm_mode}'"
                    )

        # All models must share the same wavefront_step.
        wavefront_step = self._flat_groups[0].model.wavefront_step
        for i, mg in enumerate(self._flat_groups[1:], 1):
            if mg.model.wavefront_step != wavefront_step:
                raise ValueError(
                    f"wavefront_step mismatch at flat model index {i}: "
                    f"{wavefront_step} vs {mg.model.wavefront_step}"
                )

        self.wavefront_step = wavefront_step
        self.ngroups = len(atm_groups)

        # Per-atm-group attributes.
        self.natm_per_group = [ag[0].model.natm for ag in atm_groups]
        self.atm_mode_per_group = [ag[0].model.atm_mode for ag in atm_groups]

        # Per-flat-model attributes.
        self.nstar_per_model = [mg.model.nstar for mg in self._flat_groups]
        self.npix_per_model  = [mg.model.npix  for mg in self._flat_groups]
        self.nbkg_per_model  = [mg.model.nbkg  for mg in self._flat_groups]
        self.nchi_per_model  = [n * p ** 2 for n, p in
                                zip(self.nstar_per_model, self.npix_per_model)]
        self.total_nchi = sum(self.nchi_per_model)
        self.nmodels = len(self._flat_groups)

        # Map flat model index → atm group index, and group → flat indices.
        self._model_to_group = []
        self._group_model_indices = []
        mi = 0
        for g, ag in enumerate(atm_groups):
            idxs = list(range(mi, mi + len(ag)))
            self._group_model_indices.append(idxs)
            for _ in ag:
                self._model_to_group.append(g)
                mi += 1

        # Precompute column start offsets for per-star params (fluxes/dxs/dys).
        self._perstar_col_starts = []
        col = 0
        for n in self.nstar_per_model:
            self._perstar_col_starts.append(col)
            col += 3 * n
        self._total_perstar_cols = col

        # Precompute column start offsets for atm params (one block per group).
        self._atm_col_starts = []
        for natm_g in self.natm_per_group:
            self._atm_col_starts.append(col)
            col += natm_g
        self._total_natm_cols = sum(self.natm_per_group)

        # Wavefront starts right after all atm params.
        self._wavefront_col_start = self._total_perstar_cols + self._total_natm_cols

        # Row start offsets for each flat model in the chi vector.
        self._row_starts = []
        row = 0
        for nchi in self.nchi_per_model:
            self._row_starts.append(row)
            row += nchi

    # ------------------------------------------------------------------
    # Pack / unpack
    # ------------------------------------------------------------------

    def pack_params(self, *, wavefront_params, outer_groups):
        """Pack all parameters into a flat tuple.

        Parameters
        ----------
        wavefront_params : sequence of float
            Shared wavefront parameters.
        outer_groups : list of dict
            One dict per outer atm group, each containing:

            ``'atm'`` : dict
                Keys ``'fwhm'`` (if ``atm_mode='fwhm'``) or ``'Ixx'``,
                ``'Ixy'``, ``'Iyy'`` (if ``atm_mode='ixx'``).
            ``'models'`` : list of dict
                One dict per sub-model in the group, each with keys
                ``'fluxes'``, ``'dxs'``, ``'dys'`` (sequences of float)
                and optionally ``'bkgs'`` (list of tuples; defaults to
                empty tuples).

        Returns
        -------
        params : tuple of float
        """
        params = []

        # Per-star params (fluxes, dxs, dys) in flat model order.
        for ag, og in zip(self._atm_groups, outer_groups):
            for mg, md in zip(ag, og['models']):
                params.extend(md['fluxes'])
                params.extend(md['dxs'])
                params.extend(md['dys'])

        # Atm params, one block per outer group.
        for g, (ag, og) in enumerate(zip(self._atm_groups, outer_groups)):
            atm = og['atm']
            if self.atm_mode_per_group[g] == 'fwhm':
                params.append(atm['fwhm'])
            else:
                params.extend([atm['Ixx'], atm['Ixy'], atm['Iyy']])

        # Shared wavefront.
        params.extend(wavefront_params)

        # Background params in flat model order.
        for ag, og in zip(self._atm_groups, outer_groups):
            for mg, md in zip(ag, og['models']):
                bkgs = md.get('bkgs', [()] * mg.model.nstar)
                for bkg in bkgs:
                    params.extend(bkg)

        return tuple(params)

    def unpack_params(self, params):
        """Unpack a flat parameter tuple into a nested dict.

        Returns
        -------
        dict with keys:

        ``'wavefront_params'`` : sequence of float

        ``'outer_groups'`` : list of dict, one per atm group, each with:

            ``'atm'`` : dict (``fwhm`` or ``Ixx``/``Ixy``/``Iyy``)

            ``'models'`` : list of dict, each with keys
            ``fluxes``, ``dxs``, ``dys``, ``bkgs``
        """
        idx = 0
        result = {'outer_groups': []}

        # Per-star params (fluxes, dxs, dys).
        for g, ag in enumerate(self._atm_groups):
            og_result = {'models': []}
            for mg in ag:
                n = mg.model.nstar
                fluxes = params[idx:idx + n]; idx += n
                dxs    = params[idx:idx + n]; idx += n
                dys    = params[idx:idx + n]; idx += n
                og_result['models'].append(
                    {'fluxes': fluxes, 'dxs': dxs, 'dys': dys}
                )
            result['outer_groups'].append(og_result)

        # Atm params.
        for g in range(self.ngroups):
            natm_g = self.natm_per_group[g]
            if self.atm_mode_per_group[g] == 'fwhm':
                result['outer_groups'][g]['atm'] = {'fwhm': params[idx]}
            else:
                result['outer_groups'][g]['atm'] = {
                    'Ixx': params[idx],
                    'Ixy': params[idx + 1],
                    'Iyy': params[idx + 2],
                }
            idx += natm_g

        # Wavefront params (infer count from total length).
        total_nbkg = sum(
            n * nb for n, nb in zip(self.nstar_per_model, self.nbkg_per_model)
        )
        nwf = len(params) - idx - total_nbkg
        result['wavefront_params'] = params[idx:idx + nwf]
        idx += nwf

        # Background params.
        for g, ag in enumerate(self._atm_groups):
            for m_idx, mg in enumerate(ag):
                nbkg = mg.model.nbkg
                bkgs = []
                for s in range(mg.model.nstar):
                    bkgs.append(tuple(params[idx:idx + nbkg]))
                    idx += nbkg
                result['outer_groups'][g]['models'][m_idx]['bkgs'] = bkgs

        return result

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def _sub_model_packed(self, mg, model_dict, atm_dict, wavefront_params):
        """Build packed params for a single sub-model."""
        kw = dict(
            fluxes=model_dict['fluxes'],
            dxs=model_dict['dxs'],
            dys=model_dict['dys'],
            wavefront_params=wavefront_params,
            bkgs=model_dict['bkgs'],
        )
        kw.update(atm_dict)
        return mg.model.pack_params(**kw)

    def model(self, params):
        """Compute model images for all sub-models.

        Parameters
        ----------
        params : sequence of float
            Joint packed parameters.

        Returns
        -------
        list of ndarray
            One array per flat sub-model, in flat model order.
        """
        joint_dict = self.unpack_params(params)
        wf = joint_dict['wavefront_params']
        result = []
        for ag, og_dict in zip(self._atm_groups, joint_dict['outer_groups']):
            for m_idx, mg in enumerate(ag):
                packed = self._sub_model_packed(
                    mg, og_dict['models'][m_idx], og_dict['atm'], wf
                )
                result.append(mg.model.model(**mg.model.unpack_params(packed)))
        return result

    # ------------------------------------------------------------------
    # Chi and Jacobian
    # ------------------------------------------------------------------

    def chi(self, params, data_list, var_list):
        """Compute concatenated (weighted) chi residuals for all sub-models.

        Parameters
        ----------
        params : sequence of float
            Joint packed parameters.
        data_list : list of ndarray
            One array per flat sub-model, in flat model order.
        var_list : list of float or ndarray
            Variances, one entry per flat sub-model.

        Returns
        -------
        chi : 1D ndarray
            Concatenated ``sqrt(weight) * chi_i`` for each sub-model.
        """
        joint_dict = self.unpack_params(params)
        wf = joint_dict['wavefront_params']
        chi_parts = []
        mi = 0
        for ag, og_dict in zip(self._atm_groups, joint_dict['outer_groups']):
            for m_idx, mg in enumerate(ag):
                packed = self._sub_model_packed(
                    mg, og_dict['models'][m_idx], og_dict['atm'], wf
                )
                chi_i = mg.model.chi(packed, data_list[mi], var_list[mi])
                chi_parts.append(mg.sqrt_weight * chi_i)
                mi += 1
        return np.concatenate(chi_parts)

    def jac(self, params, data_list, var_list):
        """Compute the Jacobian d(chi)/d(params).

        Uses a block-sparse finite-difference structure:

        - Per-star params (fluxes, dxs, dys) for model *m* affect only
          that model's chi rows.
        - Atm params for atm group *g* affect only chi rows of models in
          that group.
        - Wavefront params affect all chi rows (dense block).
        - Background params for model *m* affect only that model's chi rows.

        Parameters
        ----------
        params : sequence of float
        data_list, var_list : lists of arrays (same order as :meth:`chi`)

        Returns
        -------
        jac : ndarray, shape (total_nchi, nparams)
        """
        nparams = len(params)
        total_nbkg = sum(
            n * nb for n, nb in zip(self.nstar_per_model, self.nbkg_per_model)
        )
        nwf = nparams - self._total_perstar_cols - self._total_natm_cols - total_nbkg

        out = np.zeros((self.total_nchi, nparams))
        chi0 = self.chi(params, data_list, var_list)

        # Unpack and pre-build sub-model packed arrays (reused for per-star/bkg).
        joint_dict = self.unpack_params(params)
        wf = joint_dict['wavefront_params']
        packed_per_model = []
        for ag, og_dict in zip(self._atm_groups, joint_dict['outer_groups']):
            for m_idx, mg in enumerate(ag):
                packed_per_model.append(
                    np.array(
                        self._sub_model_packed(
                            mg, og_dict['models'][m_idx], og_dict['atm'], wf
                        ),
                        dtype=float,
                    )
                )

        # --- Per-star params (fluxes, dxs, dys): sparse, per model ---
        for mi, mg in enumerate(self._flat_groups):
            n        = self.nstar_per_model[mi]
            npix     = self.npix_per_model[mi]
            row0     = self._row_starts[mi]
            col0     = self._perstar_col_starts[mi]
            packed   = packed_per_model[mi]
            chi0_u   = chi0[row0:row0 + n * npix ** 2] / mg.sqrt_weight

            for offset, key, step in [(0, 'fluxes', 0.01),
                                       (n, 'dxs',    0.01),
                                       (2*n, 'dys',  0.01)]:
                d_dict = mg.model.unpack_params(packed)
                d_dict[key] = np.array(d_dict[key]) + step
                chi_p = mg.model.chi(
                    mg.model.pack_params(**d_dict), data_list[mi], var_list[mi]
                )
                for s in range(n):
                    sl  = slice(s * npix ** 2, (s + 1) * npix ** 2)
                    rsl = slice(row0 + s * npix ** 2, row0 + (s + 1) * npix ** 2)
                    out[rsl, col0 + offset + s] = (
                        mg.sqrt_weight * (chi_p[sl] - chi0_u[sl]) / step
                    )

        # --- Atm params: affect only models in the same atm group ---
        for g in range(self.ngroups):
            atm_col0 = self._atm_col_starts[g]
            for k in range(self.natm_per_group[g]):
                params1 = np.array(params, dtype=float)
                params1[atm_col0 + k] += 0.01
                chi1 = self.chi(params1, data_list, var_list)
                for mi in self._group_model_indices[g]:
                    r0    = self._row_starts[mi]
                    nchi  = self.nchi_per_model[mi]
                    out[r0:r0 + nchi, atm_col0 + k] = (
                        chi1[r0:r0 + nchi] - chi0[r0:r0 + nchi]
                    ) / 0.01

        # --- Wavefront params: dense (affects all chi rows) ---
        wf_col0 = self._wavefront_col_start
        for k in range(nwf):
            col = wf_col0 + k
            params1 = np.array(params, dtype=float)
            params1[col] += self.wavefront_step
            chi1 = self.chi(params1, data_list, var_list)
            out[:, col] = (chi1 - chi0) / self.wavefront_step

        # --- Background params: sparse, per model ---
        bkg_col = wf_col0 + nwf
        dbkg = 0.01
        for mi, mg in enumerate(self._flat_groups):
            n      = self.nstar_per_model[mi]
            npix   = self.npix_per_model[mi]
            nbkg   = self.nbkg_per_model[mi]
            row0   = self._row_starts[mi]
            packed = packed_per_model[mi]
            chi0_u = chi0[row0:row0 + n * npix ** 2] / mg.sqrt_weight

            for k in range(nbkg):
                d_dict = mg.model.unpack_params(packed)
                for s in range(n):
                    bkgj = list(d_dict['bkgs'][s])
                    bkgj[k] += dbkg
                    d_dict['bkgs'][s] = tuple(bkgj)
                chi_p = mg.model.chi(
                    mg.model.pack_params(**d_dict), data_list[mi], var_list[mi]
                )
                for s in range(n):
                    sl  = slice(s * npix ** 2, (s + 1) * npix ** 2)
                    rsl = slice(row0 + s * npix ** 2, row0 + (s + 1) * npix ** 2)
                    out[rsl, bkg_col + nbkg * s + k] = (
                        mg.sqrt_weight * (chi_p[sl] - chi0_u[sl]) / dbkg
                    )
            bkg_col += nbkg * n

        return out

    def _jac2(self, params, data_list, var_list):
        """Naive column-by-column Jacobian for validation."""
        nparams = len(params)
        total_nbkg = sum(
            n * nb for n, nb in zip(self.nstar_per_model, self.nbkg_per_model)
        )
        nwf = nparams - self._total_perstar_cols - self._total_natm_cols - total_nbkg

        step = []
        for n in self.nstar_per_model:
            step += [0.01] * (3 * n)           # fluxes, dxs, dys
        for natm_g in self.natm_per_group:
            step += [0.01] * natm_g            # atm
        step += [self.wavefront_step] * nwf    # wavefront
        for n, nb in zip(self.nstar_per_model, self.nbkg_per_model):
            step += [0.01] * (n * nb)          # bkgs

        out = np.empty((self.total_nchi, nparams))
        chi0 = self.chi(params, data_list, var_list)
        for i, h in enumerate(step):
            params1 = np.array(params, dtype=float)
            params1[i] += h
            chi1 = self.chi(params1, data_list, var_list)
            out[:, i] = (chi1 - chi0) / h
        return out


class DZMultiGroupJointModel(MultiGroupJointModel):
    """MultiGroupJointModel using double Zernike wavefront parameterization.

    All sub-models must be :class:`DZMultiDonutModel` or
    :class:`DZMultiSpotModel` with identical ``dz_terms``.
    """

    def __init__(self, atm_groups):
        flat = [mg for ag in atm_groups for mg in ag]
        for i, mg in enumerate(flat):
            if not isinstance(mg.model, (DZMultiDonutModel, DZMultiSpotModel)):
                raise TypeError(
                    f"All models must be DZMultiDonutModel or DZMultiSpotModel; "
                    f"flat model index {i} is {type(mg.model).__name__}"
                )
        dz_terms_ref = tuple(flat[0].model.dz_terms)
        for i, mg in enumerate(flat[1:], 1):
            if tuple(mg.model.dz_terms) != dz_terms_ref:
                raise ValueError(f"dz_terms mismatch at flat model index {i}")
        super().__init__(atm_groups)
        self.dz_terms = flat[0].model.dz_terms
        self.nwavefront = len(self.dz_terms)


class DZBasisMultiGroupJointModel(MultiGroupJointModel):
    """MultiGroupJointModel using sensitivity-matrix wavefront parameterization.

    All sub-models must be :class:`DZBasisMultiDonutModel` or
    :class:`DZBasisMultiSpotModel` with identical ``sensitivity`` matrices.
    """

    def __init__(self, atm_groups):
        flat = [mg for ag in atm_groups for mg in ag]
        for i, mg in enumerate(flat):
            if not isinstance(
                mg.model, (DZBasisMultiDonutModel, DZBasisMultiSpotModel)
            ):
                raise TypeError(
                    f"All models must be DZBasisMultiDonutModel or "
                    f"DZBasisMultiSpotModel; flat model index {i} is "
                    f"{type(mg.model).__name__}"
                )
        sensitivity_ref = flat[0].model.sensitivity
        for i, mg in enumerate(flat[1:], 1):
            if not np.array_equal(mg.model.sensitivity, sensitivity_ref):
                raise ValueError(
                    f"sensitivity mismatch at flat model index {i}"
                )
        super().__init__(atm_groups)
        self.sensitivity = flat[0].model.sensitivity
        self.nwavefront = flat[0].model.nmode


class JointModel:
    """Joint donut + spot model for simultaneous fitting.

    Composes a donut sub-model and a spot sub-model that share atmospheric and
    wavefront parameters.  Per-star parameters (fluxes, dxs, dys, bkgs) are
    independent between the two modalities.

    Parameters
    ----------
    donut_model : BaseMultiDonutModel subclass
        Pre-constructed donut fitter.
    spot_model : BaseMultiSpotModel subclass
        Pre-constructed spot fitter.
    spot_weight : float, optional
        Relative weight of spot chi residuals.  The spot chi values are
        multiplied by sqrt(spot_weight) so that the cost contribution is
        scaled by spot_weight.  Default 1.0.
    """
    def __init__(self, donut_model, spot_model, spot_weight=1.0):
        if type(self) is JointModel:
            warnings.warn(
                "JointModel is deprecated; use MultiGroupJointModel instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if donut_model.atm_mode != spot_model.atm_mode:
            raise ValueError(
                f"atm_mode mismatch: donut={donut_model.atm_mode}, "
                f"spot={spot_model.atm_mode}"
            )
        if donut_model.wavefront_step != spot_model.wavefront_step:
            raise ValueError(
                f"wavefront_step mismatch: donut={donut_model.wavefront_step}, "
                f"spot={spot_model.wavefront_step}"
            )

        self.donut_model = donut_model
        self.spot_model = spot_model
        self.spot_weight = spot_weight
        self.sqrt_spot_weight = np.sqrt(spot_weight)

        self.nd = donut_model.nstar
        self.ns = spot_model.nstar
        self.atm_mode = donut_model.atm_mode
        self.natm = donut_model.natm
        self.nbkg_d = donut_model.nbkg
        self.nbkg_s = spot_model.nbkg
        self.wavefront_step = donut_model.wavefront_step

        self.npix_d = donut_model.npix
        self.npix_s = spot_model.npix
        self.nchi_d = self.nd * self.npix_d ** 2
        self.nchi_s = self.ns * self.npix_s ** 2

    def pack_params(
        self, *,
        d_fluxes, d_dxs, d_dys,
        s_fluxes, s_dxs, s_dys,
        fwhm=None, Ixx=None, Ixy=None, Iyy=None,
        wavefront_params,
        d_bkgs=None, s_bkgs=None
    ):
        """Pack joint parameters into a single tuple.

        Layout: [d_fluxes | d_dxs | d_dys | s_fluxes | s_dxs | s_dys |
                 atm(natm) | wavefront | d_bkgs | s_bkgs]
        """
        if d_bkgs is None:
            d_bkgs = [()] * self.nd
        if s_bkgs is None:
            s_bkgs = [()] * self.ns
        params = []
        params.extend(d_fluxes)
        params.extend(d_dxs)
        params.extend(d_dys)
        params.extend(s_fluxes)
        params.extend(s_dxs)
        params.extend(s_dys)
        if self.atm_mode == 'fwhm':
            params.append(fwhm)
        else:
            params.extend([Ixx, Ixy, Iyy])
        params.extend(wavefront_params)
        for bkg in d_bkgs:
            params.extend(bkg)
        for bkg in s_bkgs:
            params.extend(bkg)
        return tuple(params)

    def unpack_params(self, params):
        """Unpack joint parameters from optimization tuple.

        Returns
        -------
        dict with keys:
            d_fluxes, d_dxs, d_dys, s_fluxes, s_dxs, s_dys,
            fwhm or (Ixx, Ixy, Iyy), wavefront_params, d_bkgs, s_bkgs
        """
        nd = self.nd
        ns = self.ns
        natm = self.natm

        idx = 0
        d_fluxes = params[idx:idx+nd]; idx += nd
        d_dxs = params[idx:idx+nd]; idx += nd
        d_dys = params[idx:idx+nd]; idx += nd
        s_fluxes = params[idx:idx+ns]; idx += ns
        s_dxs = params[idx:idx+ns]; idx += ns
        s_dys = params[idx:idx+ns]; idx += ns

        out = dict(
            d_fluxes=d_fluxes, d_dxs=d_dxs, d_dys=d_dys,
            s_fluxes=s_fluxes, s_dxs=s_dxs, s_dys=s_dys,
        )

        if self.atm_mode == 'fwhm':
            out['fwhm'] = params[idx]; idx += 1
        else:
            out['Ixx'] = params[idx]; idx += 1
            out['Ixy'] = params[idx]; idx += 1
            out['Iyy'] = params[idx]; idx += 1

        # Wavefront params: everything between atm and bkgs
        nbkg_total = self.nbkg_d * nd + self.nbkg_s * ns
        nwf = len(params) - idx - nbkg_total
        out['wavefront_params'] = params[idx:idx+nwf]; idx += nwf

        d_bkgs = []
        for i in range(nd):
            d_bkgs.append(tuple(params[idx:idx+self.nbkg_d]))
            idx += self.nbkg_d
        out['d_bkgs'] = d_bkgs

        s_bkgs = []
        for i in range(ns):
            s_bkgs.append(tuple(params[idx:idx+self.nbkg_s]))
            idx += self.nbkg_s
        out['s_bkgs'] = s_bkgs

        return out

    def _donut_packed(self, joint_dict):
        """Build packed params for the donut sub-model from joint dict."""
        kw = dict(
            fluxes=joint_dict['d_fluxes'],
            dxs=joint_dict['d_dxs'],
            dys=joint_dict['d_dys'],
            wavefront_params=joint_dict['wavefront_params'],
            bkgs=joint_dict['d_bkgs'],
        )
        if self.atm_mode == 'fwhm':
            kw['fwhm'] = joint_dict['fwhm']
        else:
            kw['Ixx'] = joint_dict['Ixx']
            kw['Ixy'] = joint_dict['Ixy']
            kw['Iyy'] = joint_dict['Iyy']
        return self.donut_model.pack_params(**kw)

    def _spot_packed(self, joint_dict):
        """Build packed params for the spot sub-model from joint dict."""
        kw = dict(
            fluxes=joint_dict['s_fluxes'],
            dxs=joint_dict['s_dxs'],
            dys=joint_dict['s_dys'],
            wavefront_params=joint_dict['wavefront_params'],
            bkgs=joint_dict['s_bkgs'],
        )
        if self.atm_mode == 'fwhm':
            kw['fwhm'] = joint_dict['fwhm']
        else:
            kw['Ixx'] = joint_dict['Ixx']
            kw['Ixy'] = joint_dict['Ixy']
            kw['Iyy'] = joint_dict['Iyy']
        return self.spot_model.pack_params(**kw)

    def model(self, **kwargs):
        """Compute model images for both donuts and spots.

        Parameters
        ----------
        **kwargs : as returned by unpack_params

        Returns
        -------
        donut_imgs : array, shape (nd, npix_d, npix_d)
        spot_imgs : array, shape (ns, npix_s, npix_s)
        """
        d_packed = self._donut_packed(kwargs)
        s_packed = self._spot_packed(kwargs)
        d_imgs = self.donut_model.model(
            **self.donut_model.unpack_params(d_packed)
        )
        s_imgs = self.spot_model.model(
            **self.spot_model.unpack_params(s_packed)
        )
        return d_imgs, s_imgs

    def chi(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Compute joint chi residuals.

        Parameters
        ----------
        params : sequence of float
            Joint packed parameters.
        donut_data : array, shape (nd, npix_d, npix_d)
        donut_vars : sequence of float or array
        spot_data : array, shape (ns, npix_s, npix_s)
        spot_vars : sequence of float or array

        Returns
        -------
        chi : array of float
            Concatenated [donut_chi, sqrt_spot_weight * spot_chi].
        """
        joint_dict = self.unpack_params(params)
        d_packed = self._donut_packed(joint_dict)
        s_packed = self._spot_packed(joint_dict)
        d_chi = self.donut_model.chi(d_packed, donut_data, donut_vars)
        s_chi = self.spot_model.chi(s_packed, spot_data, spot_vars)
        return np.concatenate([d_chi, self.sqrt_spot_weight * s_chi])

    def jac(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Compute joint jacobian d(chi)/d(param).

        Uses block-sparse structure: donut per-star params only affect donut
        rows, spot per-star params only affect spot rows, shared atm+wavefront
        params affect all rows.

        Parameters
        ----------
        params : sequence of float
        donut_data, donut_vars, spot_data, spot_vars : arrays

        Returns
        -------
        jac : array of float, shape (nchi_d + nchi_s, nparams)
        """
        nd = self.nd
        ns = self.ns
        natm = self.natm
        nbkg_d = self.nbkg_d
        nbkg_s = self.nbkg_s
        nchi_d = self.nchi_d
        nchi_s = self.nchi_s
        nparams = len(params)

        # Compute the number of wavefront params
        nperstar = 3*nd + 3*ns
        nbkg_total = nbkg_d*nd + nbkg_s*ns
        nwf = nparams - nperstar - natm - nbkg_total

        out = np.zeros((nchi_d + nchi_s, nparams))
        chi0 = self.chi(params, donut_data, donut_vars, spot_data, spot_vars)
        chi0_d = chi0[:nchi_d]
        chi0_s = chi0[nchi_d:]

        joint_dict = self.unpack_params(params)
        d_packed = np.array(self._donut_packed(joint_dict), dtype=float)
        s_packed = np.array(self._spot_packed(joint_dict), dtype=float)

        # --- Donut per-star sparse params (d_fluxes, d_dxs, d_dys) ---
        # These only affect donut chi rows.
        # d_fluxes columns: [0, nd)
        npix_d = self.npix_d
        dflux = 0.01
        d_dict = self.donut_model.unpack_params(d_packed)
        d_dict["fluxes"] = np.array(d_dict["fluxes"]) + dflux
        chi_d = self.donut_model.chi(
            self.donut_model.pack_params(**d_dict), donut_data, donut_vars
        )
        for i in range(nd):
            s = slice(i*npix_d**2, (i+1)*npix_d**2)
            out[s, i] = (chi_d[s] - chi0_d[s]) / dflux

        # d_dxs columns: [nd, 2*nd)
        dx = 0.01
        d_dict = self.donut_model.unpack_params(d_packed)
        d_dict["dxs"] = np.array(d_dict["dxs"]) + dx
        chi_d = self.donut_model.chi(
            self.donut_model.pack_params(**d_dict), donut_data, donut_vars
        )
        for i in range(nd):
            s = slice(i*npix_d**2, (i+1)*npix_d**2)
            out[s, nd+i] = (chi_d[s] - chi0_d[s]) / dx

        # d_dys columns: [2*nd, 3*nd)
        dy = 0.01
        d_dict = self.donut_model.unpack_params(d_packed)
        d_dict["dys"] = np.array(d_dict["dys"]) + dy
        chi_d = self.donut_model.chi(
            self.donut_model.pack_params(**d_dict), donut_data, donut_vars
        )
        for i in range(nd):
            s = slice(i*npix_d**2, (i+1)*npix_d**2)
            out[s, 2*nd+i] = (chi_d[s] - chi0_d[s]) / dy

        # --- Spot per-star sparse params (s_fluxes, s_dxs, s_dys) ---
        # These only affect spot chi rows.
        # s_fluxes columns: [3*nd, 3*nd+ns)
        npix_s = self.npix_s
        s_dict = self.spot_model.unpack_params(s_packed)
        s_dict["fluxes"] = np.array(s_dict["fluxes"]) + dflux
        chi_s = self.spot_model.chi(
            self.spot_model.pack_params(**s_dict), spot_data, spot_vars
        )
        for i in range(ns):
            s = slice(i*npix_s**2, (i+1)*npix_s**2)
            out[nchi_d+s.start:nchi_d+s.stop, 3*nd+i] = (
                self.sqrt_spot_weight * (chi_s[s] - chi0_s[s] / self.sqrt_spot_weight)
            ) / dflux

        # s_dxs columns: [3*nd+ns, 3*nd+2*ns)
        s_dict = self.spot_model.unpack_params(s_packed)
        s_dict["dxs"] = np.array(s_dict["dxs"]) + dx
        chi_s = self.spot_model.chi(
            self.spot_model.pack_params(**s_dict), spot_data, spot_vars
        )
        for i in range(ns):
            s = slice(i*npix_s**2, (i+1)*npix_s**2)
            out[nchi_d+s.start:nchi_d+s.stop, 3*nd+ns+i] = (
                self.sqrt_spot_weight * (chi_s[s] - chi0_s[s] / self.sqrt_spot_weight)
            ) / dx

        # s_dys columns: [3*nd+2*ns, 3*nd+3*ns)
        s_dict = self.spot_model.unpack_params(s_packed)
        s_dict["dys"] = np.array(s_dict["dys"]) + dy
        chi_s = self.spot_model.chi(
            self.spot_model.pack_params(**s_dict), spot_data, spot_vars
        )
        for i in range(ns):
            s = slice(i*npix_s**2, (i+1)*npix_s**2)
            out[nchi_d+s.start:nchi_d+s.stop, 3*nd+2*ns+i] = (
                self.sqrt_spot_weight * (chi_s[s] - chi0_s[s] / self.sqrt_spot_weight)
            ) / dy

        # --- Shared atm + wavefront params (dense: affect all rows) ---
        shared_start = 3*nd + 3*ns
        for k in range(natm):
            params1 = np.array(params, dtype=float)
            params1[shared_start + k] += 0.01
            chi1 = self.chi(
                params1, donut_data, donut_vars, spot_data, spot_vars
            )
            out[:, shared_start + k] = (chi1 - chi0) / 0.01

        for k in range(nwf):
            col = shared_start + natm + k
            params1 = np.array(params, dtype=float)
            params1[col] += self.wavefront_step
            chi1 = self.chi(
                params1, donut_data, donut_vars, spot_data, spot_vars
            )
            out[:, col] = (chi1 - chi0) / self.wavefront_step

        # --- Donut bkg params (sparse: only donut rows) ---
        dbkg = 0.01
        bkg_d_start = shared_start + natm + nwf
        for k in range(nbkg_d):
            d_dict = self.donut_model.unpack_params(d_packed)
            for j in range(nd):
                bkgj = list(d_dict["bkgs"][j])
                bkgj[k] += dbkg
                d_dict["bkgs"][j] = tuple(bkgj)
            chi_d = self.donut_model.chi(
                self.donut_model.pack_params(**d_dict), donut_data, donut_vars
            )
            for j in range(nd):
                s = slice(j*npix_d**2, (j+1)*npix_d**2)
                out[s, bkg_d_start + nbkg_d*j + k] = (
                    chi_d[s] - chi0_d[s]
                ) / dbkg

        # --- Spot bkg params (sparse: only spot rows) ---
        bkg_s_start = bkg_d_start + nbkg_d * nd
        for k in range(nbkg_s):
            s_dict = self.spot_model.unpack_params(s_packed)
            for j in range(ns):
                bkgj = list(s_dict["bkgs"][j])
                bkgj[k] += dbkg
                s_dict["bkgs"][j] = tuple(bkgj)
            chi_s = self.spot_model.chi(
                self.spot_model.pack_params(**s_dict), spot_data, spot_vars
            )
            for j in range(ns):
                s = slice(j*npix_s**2, (j+1)*npix_s**2)
                chi0_sj = chi0_s[s] / self.sqrt_spot_weight
                out[nchi_d+s.start:nchi_d+s.stop, bkg_s_start + nbkg_s*j + k] = (
                    self.sqrt_spot_weight * (chi_s[s] - chi0_sj)
                ) / dbkg

        return out

    def _jac2(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Naive column-by-column jacobian for validation."""
        nd = self.nd
        ns = self.ns
        natm = self.natm
        nbkg_d = self.nbkg_d
        nbkg_s = self.nbkg_s
        nparams = len(params)

        nperstar = 3*nd + 3*ns
        nbkg_total = nbkg_d*nd + nbkg_s*ns
        nwf = nparams - nperstar - natm - nbkg_total

        nchi = self.nchi_d + self.nchi_s
        out = np.empty((nchi, nparams))

        step = [0.01]*(3*nd)  # d_fluxes, d_dxs, d_dys
        step += [0.01]*(3*ns)  # s_fluxes, s_dxs, s_dys
        step += [0.01]*natm
        step += [self.wavefront_step]*nwf
        step += [0.01]*(nbkg_d*nd)
        step += [0.01]*(nbkg_s*ns)

        chi0 = self.chi(params, donut_data, donut_vars, spot_data, spot_vars)
        for i, h in enumerate(step):
            params1 = np.array(params, dtype=float)
            params1[i] += h
            chi1 = self.chi(
                params1, donut_data, donut_vars, spot_data, spot_vars
            )
            out[:, i] = (chi1 - chi0) / h

        return out


class DZJointModel(JointModel):
    """Joint donut + spot model using double Zernike parameterization.

    Both sub-models must be DZMultiDonutModel and DZMultiSpotModel with
    identical dz_terms.

    Parameters
    ----------
    donut_model : DZMultiDonutModel
        Pre-constructed donut fitter.
    spot_model : DZMultiSpotModel
        Pre-constructed spot fitter.  Must have the same ``dz_terms`` as
        ``donut_model``.
    spot_weight : float, optional
        Relative weight of spot chi residuals.  Default 1.0.
    """
    def __init__(self, donut_model, spot_model, spot_weight=1.0):
        warnings.warn(
            "DZJointModel is deprecated; use DZMultiGroupJointModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(donut_model, DZMultiDonutModel):
            raise TypeError("donut_model must be a DZMultiDonutModel")
        if not isinstance(spot_model, DZMultiSpotModel):
            raise TypeError("spot_model must be a DZMultiSpotModel")
        if tuple(donut_model.dz_terms) != tuple(spot_model.dz_terms):
            raise ValueError(
                "dz_terms mismatch between donut and spot models"
            )
        super().__init__(donut_model, spot_model, spot_weight=spot_weight)
        self.dz_terms = donut_model.dz_terms
        self.nwavefront = len(self.dz_terms)


class DZBasisJointModel(JointModel):
    """Joint donut + spot model using sensitivity matrix parameterization.

    Both sub-models must be DZBasisMultiDonutModel and DZBasisMultiSpotModel
    with identical sensitivity matrices.

    Parameters
    ----------
    donut_model : DZBasisMultiDonutModel
        Pre-constructed donut fitter.
    spot_model : DZBasisMultiSpotModel
        Pre-constructed spot fitter.  Must have the same ``sensitivity``
        matrix as ``donut_model``.
    spot_weight : float, optional
        Relative weight of spot chi residuals.  Default 1.0.
    """
    def __init__(self, donut_model, spot_model, spot_weight=1.0):
        warnings.warn(
            "DZBasisJointModel is deprecated; use DZBasisMultiGroupJointModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(donut_model, DZBasisMultiDonutModel):
            raise TypeError("donut_model must be a DZBasisMultiDonutModel")
        if not isinstance(spot_model, DZBasisMultiSpotModel):
            raise TypeError("spot_model must be a DZBasisMultiSpotModel")
        if not np.array_equal(donut_model.sensitivity, spot_model.sensitivity):
            raise ValueError(
                "sensitivity mismatch between donut and spot models"
            )
        super().__init__(donut_model, spot_model, spot_weight=spot_weight)
        self.sensitivity = donut_model.sensitivity
        self.nwavefront = donut_model.nmode
