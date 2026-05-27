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
        if len(outer_groups) != self.ngroups:
            raise ValueError(
                f"outer_groups has {len(outer_groups)} entries; "
                f"expected {self.ngroups}"
            )
        for g, (ag, og) in enumerate(zip(self._atm_groups, outer_groups)):
            if len(og['models']) != len(ag):
                raise ValueError(
                    f"outer_groups[{g}]['models'] has {len(og['models'])} "
                    f"entries; expected {len(ag)}"
                )

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
        # Explicit base-class call: bypass any subclass unpack_params override
        # (e.g. JointModel's backwards-compat shim) so we always get the
        # nested-dict form this method expects.
        joint_dict = MultiGroupJointModel.unpack_params(self, params)
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
        # Use the base-class unpack explicitly so that subclass overrides of
        # unpack_params (e.g. JointModel's backwards-compat shim) don't break
        # the internal nested-dict contract.
        joint_dict = MultiGroupJointModel.unpack_params(self, params)
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
        # Explicit base-class calls below: same rationale as in chi() — bypass
        # any subclass override of chi/unpack_params to keep the internal
        # nested-dict contract intact.
        chi0 = MultiGroupJointModel.chi(self, params, data_list, var_list)

        # Unpack and pre-build sub-model packed arrays (reused for per-star/bkg).
        joint_dict = MultiGroupJointModel.unpack_params(self, params)
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

            # All n stars are perturbed simultaneously (one chi call per
            # param type rather than per star).  This is valid because each
            # star's chi rows depend only on that star's own per-star params,
            # so the cross-star perturbations do not contaminate the slice
            # extracted for star s.
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
                chi1 = MultiGroupJointModel.chi(self, params1, data_list, var_list)
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
            chi1 = MultiGroupJointModel.chi(self, params1, data_list, var_list)
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
        chi0 = MultiGroupJointModel.chi(self, params, data_list, var_list)
        for i, h in enumerate(step):
            params1 = np.array(params, dtype=float)
            params1[i] += h
            chi1 = MultiGroupJointModel.chi(self, params1, data_list, var_list)
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
        # super().__init__ validates that atm_groups is non-empty and each group
        # is non-empty, so flat[0] is safe to access after this call.
        super().__init__(atm_groups)
        dz_terms_ref = tuple(flat[0].model.dz_terms)
        for i, mg in enumerate(flat[1:], 1):
            if tuple(mg.model.dz_terms) != dz_terms_ref:
                raise ValueError(f"dz_terms mismatch at flat model index {i}")
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
        # super().__init__ validates that atm_groups is non-empty and each group
        # is non-empty, so flat[0] is safe to access after this call.
        super().__init__(atm_groups)
        sensitivity_ref = flat[0].model.sensitivity
        for i, mg in enumerate(flat[1:], 1):
            if not np.array_equal(mg.model.sensitivity, sensitivity_ref):
                raise ValueError(
                    f"sensitivity mismatch at flat model index {i}"
                )
        self.sensitivity = flat[0].model.sensitivity
        self.nwavefront = flat[0].model.nmode


class JointModel(MultiGroupJointModel):
    """Joint donut + spot model for simultaneous fitting.

    .. deprecated::
        Use :class:`MultiGroupJointModel` instead.

    Thin backwards-compatible facade over :class:`MultiGroupJointModel`.
    Composes one donut sub-model and one spot sub-model that share a single
    atmospheric kernel and wavefront parameterization.  The parameter vector
    layout is identical to the old standalone implementation.

    Parameters
    ----------
    donut_model : BaseMultiDonutModel subclass
        Pre-constructed donut fitter.
    spot_model : BaseMultiSpotModel subclass
        Pre-constructed spot fitter.
    spot_weight : float, optional
        Relative weight of spot chi residuals.  Default 1.0.
    """

    def __init__(self, donut_model, spot_model, spot_weight=1.0):
        # Use `type(self) is JointModel` rather than isinstance so that
        # DZJointModel/DZBasisJointModel (which issue their own deprecation
        # warnings before calling super()) don't trigger a second warning here.
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
        super().__init__([[ModelGroup(donut_model, 1.0),
                           ModelGroup(spot_model, spot_weight)]])
        # Legacy attributes preserved for backwards compatibility.
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
        self.npix_d = donut_model.npix
        self.npix_s = spot_model.npix
        self.nchi_d = self.nd * self.npix_d ** 2
        self.nchi_s = self.ns * self.npix_s ** 2

    # ------------------------------------------------------------------
    # Backwards-compatible API (old positional signatures)
    # ------------------------------------------------------------------

    def pack_params(
        self, *,
        d_fluxes, d_dxs, d_dys,
        s_fluxes, s_dxs, s_dys,
        fwhm=None, Ixx=None, Ixy=None, Iyy=None,
        wavefront_params,
        d_bkgs=None, s_bkgs=None,
    ):
        """Pack joint parameters into a single tuple (old API)."""
        if d_bkgs is None:
            d_bkgs = [()] * self.nd
        if s_bkgs is None:
            s_bkgs = [()] * self.ns
        if self.atm_mode == 'fwhm':
            atm = {'fwhm': fwhm}
        else:
            atm = {'Ixx': Ixx, 'Ixy': Ixy, 'Iyy': Iyy}
        return MultiGroupJointModel.pack_params(
            self,
            wavefront_params=wavefront_params,
            outer_groups=[{
                'atm': atm,
                'models': [
                    {'fluxes': d_fluxes, 'dxs': d_dxs, 'dys': d_dys,
                     'bkgs': d_bkgs},
                    {'fluxes': s_fluxes, 'dxs': s_dxs, 'dys': s_dys,
                     'bkgs': s_bkgs},
                ],
            }],
        )

    def unpack_params(self, params):
        """Unpack joint parameters (old API).

        Returns
        -------
        dict with keys:
            d_fluxes, d_dxs, d_dys, s_fluxes, s_dxs, s_dys,
            fwhm or (Ixx, Ixy, Iyy), wavefront_params, d_bkgs, s_bkgs
        """
        new = MultiGroupJointModel.unpack_params(self, params)
        og = new['outer_groups'][0]
        dm, sm = og['models'][0], og['models'][1]
        out = dict(
            d_fluxes=dm['fluxes'], d_dxs=dm['dxs'], d_dys=dm['dys'],
            s_fluxes=sm['fluxes'], s_dxs=sm['dxs'], s_dys=sm['dys'],
            wavefront_params=new['wavefront_params'],
            d_bkgs=dm['bkgs'], s_bkgs=sm['bkgs'],
        )
        out.update(og['atm'])
        return out

    def model(self, **kwargs):
        """Compute model images for both donuts and spots (old API).

        Parameters
        ----------
        **kwargs : as returned by :meth:`unpack_params`

        Returns
        -------
        donut_imgs : ndarray, shape (nd, npix_d, npix_d)
        spot_imgs  : ndarray, shape (ns, npix_s, npix_s)
        """
        packed = self.pack_params(**kwargs)
        imgs = MultiGroupJointModel.model(self, packed)
        return imgs[0], imgs[1]

    def chi(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Compute joint chi residuals (old API).

        Returns
        -------
        chi : array of float
            Concatenated [donut_chi, sqrt_spot_weight * spot_chi].
        """
        return MultiGroupJointModel.chi(
            self, params, [donut_data, spot_data], [donut_vars, spot_vars]
        )

    def jac(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Compute joint jacobian (old API).

        Returns
        -------
        jac : ndarray, shape (nchi_d + nchi_s, nparams)
        """
        return MultiGroupJointModel.jac(
            self, params, [donut_data, spot_data], [donut_vars, spot_vars]
        )

    def _jac2(self, params, donut_data, donut_vars, spot_data, spot_vars):
        """Naive column-by-column jacobian for validation (old API)."""
        return MultiGroupJointModel._jac2(
            self, params, [donut_data, spot_data], [donut_vars, spot_vars]
        )


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
