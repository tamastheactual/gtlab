# Publishing `gtlab`

Releases use **PyPI trusted publishing** (OIDC) via GitHub Actions - no API
tokens are stored anywhere. The flow is: register a one-time "pending publisher"
on the index website, then trigger the workflow.

## One-time setup

### 1. TestPyPI (dry run)

1. Create an account at <https://test.pypi.org/account/register/>.
2. Go to <https://test.pypi.org/manage/account/publishing/> and add a
   **pending publisher** with exactly:
   - PyPI Project Name: `gtlab`
   - Owner: `tamastheactual`
   - Repository name: `gtlab`
   - Workflow name: `publish.yml`
   - Environment name: `testpypi`

### 2. PyPI (real)

1. Create an account at <https://pypi.org/account/register/>.
2. Go to <https://pypi.org/manage/account/publishing/> and add a pending
   publisher with the same values, except:
   - Environment name: `pypi`

(Optional but recommended: in the GitHub repo, create the `testpypi` and `pypi`
**environments** under Settings → Environments to gate releases.)

## Releasing

### Dry run to TestPyPI

GitHub → **Actions** tab → **Publish** workflow → **Run workflow** →
target `testpypi`. Then verify:

```bash
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ gtlab
```

(The extra index is needed because TestPyPI doesn't host numpy/scipy/etc.)

### Real release to PyPI

Bump the version in `pyproject.toml`, commit, then tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag push runs the `pypi` job automatically. After it finishes:

```bash
pip install gtlab
```

## Notes

- A given version can be uploaded **only once** per index - bump the version
  for any re-release.
- `python -m build && twine check dist/*` reproduces the workflow's build step
  locally if you want to inspect artifacts first.
