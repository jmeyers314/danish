# Documentation workflow

Danish uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
with [mike](https://github.com/jimporter/mike) for versioned documentation
hosted on GitHub Pages at https://jmeyers314.github.io/danish/.

## Lifecycle

| Situation | Action |
|-----------|--------|
| Developing new docs | Push to `doc` branch → CI deploys preview to `/dev/` |
| Ready to ship | Open PR: `doc` → `main` (normal review + merge) |
| Cutting a release | Tag + publish GitHub Release → CI auto-deploys to `/X.Y.Z/` + `latest` |
| Fix docs for a past release | Edit on `doc`, then run `mike deploy --push X.Y.Z` |
| Fix docs for latest release | Same, but also update the alias: `mike deploy --push X.Y.Z latest` |

## Manually deploying a specific version

From the GitHub Actions UI:
1. Go to **Actions → Deploy docs to GitHub Pages → Run workflow**
2. Set `version` (e.g. `1.1.0`) and optionally `alias` (e.g. `latest`)
3. Click **Run workflow**

Or locally

    (mike deploy --push X.Y.Z)
    # To also update the 'latest' alias:
    (mike deploy --push --update-aliases X.Y.Z latest)

If the push fails with "fetch first" or the fetch itself is rejected with
"non-fast-forward", CI has pushed to `gh-pages` since your local copy was
last synced.  Delete the local branch and re-fetch before re-running mike:

    git branch -D gh-pages
    git fetch origin gh-pages:gh-pages
    (mike deploy --push --update-aliases X.Y.Z latest)

## The `doc` branch

The `doc` branch is a staging area for documentation work in progress.
It does not need to be kept in sync with `main` at all times.  When a
doc update is ready, open a PR from `doc` into `main` and merge normally.

Once all doc work is merged into `main`, `doc` should contain nothing that
isn't already in `main`.  When resuming doc work after a gap, reset `doc`
to the tip of `main` before starting:

    git checkout doc
    git reset --hard main
