# Releasing

Releases are published to GitHub Packages (`maven.pkg.github.com/ponder-lab/ML`), including the consumer-facing fat JAR `com.ibm.wala.cast.python.ml-X.Y.Z-fat.jar`. The mechanism is `maven-release-plugin`: it pushes two version-bump commits and a plain-semver tag to `master`, and the tag push drives CI's deploy gate (publish to GitHub Packages) and `release-upload` job (create the GitHub Release with generated notes and attach the fat JAR; the release is marked a pre-release only when the tag carries a `-` suffix). The normal path needs no manual `mvn deploy` or `gh release create` step (a manual-deploy fallback for CI failures is in [Troubleshooting](#troubleshooting) below).

## One-Click Release (Preferred)

Dispatch the **Cut release** workflow (`.github/workflows/release.yml`), which runs `maven-release-plugin` on a runner — no local toolchain, flags, or credentials. From the Actions UI: **Actions → Cut release → Run workflow**, select branch `master`, and enter the versions. Or from the CLI:

```bash
gh workflow run release.yml --repo ponder-lab/ML --ref master \
	-f releaseVersion=0.48.0 -f developmentVersion=0.48.1-SNAPSHOT
```

`releaseVersion` is plain semver (`X.Y.Z`); `developmentVersion` is the next snapshot (`X.Y.W-SNAPSHOT`). The workflow validates the inputs, tests the release version before tagging, pushes the commits and tag, and the tag-push CI then deploys and creates the release. Confirm the published version afterward:

```bash
gh api "/orgs/ponder-lab/packages/maven/com.ibm.wala.com.ibm.wala.cast.python.ml/versions" --jq '.[].name' | head
```

**One-time setup:** the workflow needs a `RELEASE_PAT` repository secret — a *classic* PAT (fine-grained PATs do not authenticate to `maven.pkg.github.com`) with `repo` scope, owned by an OrganizationAdmin so its push bypasses the `master` branch protection (`github-actions[bot]` is not a bypass actor; the default `GITHUB_TOKEN` push would be rejected).

## Manual Release (Fallback)

If you need to cut a release locally, the prerequisites are the local-dependency installs (Jython 3 at `0.0.2`, `cast.lsp` at `0.0.1`; `release:prepare` rejects SNAPSHOT dependencies—see **Building** in [`CONTRIBUTING.md`](CONTRIBUTING.md)) and push access to `master` (the version-bump commits push directly, bypassing the pull-request requirement via the OrganizationAdmin bypass, sanctioned per [wala/ML#457](https://github.com/wala/ML/issues/457)).

1. From a clean checkout of `master`:

	```bash
	mvn release:clean -B
	mvn release:prepare -B \
		-DreleaseVersion=X.Y.Z \
		-DdevelopmentVersion=X.Y.W-SNAPSHOT
	```

	No other flags are needed: the tag format is pinned to plain `X.Y.Z` ([wala/ML#560](https://github.com/wala/ML/issues/560)) and the local dependencies are installed at release coordinates, so `-Dtag`, `-DignoreSnapshots`, and `-DallowTimestampedSnapshots` are unnecessary. `release:prepare` builds and tests the release version, sets it, commits, tags, sets the next development version, commits, and pushes everything to `master`.

1. The tag push triggers CI's deploy gate ([wala/ML#421](https://github.com/wala/ML/issues/421)), which requires a semver-shaped tag (optionally `v`-prefixed, optionally with a `-` pre-release suffix) that is an ancestor of `master`, publishing the artifacts and creating the release, exactly as in the one-click flow. Confirm with the `gh api` command above.

## Troubleshooting

- **The tag pushed but the deploy did not fire** (rare — e.g., the tag's commit predates a workflow fix). Re-trigger the tag-push event: `git push --delete origin X.Y.Z && git push origin X.Y.Z`. ([wala/ML#454](https://github.com/wala/ML/issues/454) tracks a `workflow_dispatch` recovery for this.)
- **The pre-commit hook rejects `release:prepare`'s commit** ("files were modified by this hook"), in the manual flow only — the workflow runner has no hook. This was a Spotless / `maven-release-plugin` self-closing-tag conflict, fixed in [wala/ML#566](https://github.com/wala/ML/issues/566) (`sortPom`'s `spaceBeforeCloseEmptyElement`). If it ever recurs, run the release with git hooks disabled: `git config core.hooksPath "$(mktemp -d)"`, run `release:prepare`, then `git config --unset core.hooksPath`.
- **Manual-deploy fallback.** If CI cannot deploy a tag at all, deploy from a clean local build: check out the release commit, `mvn spotless:apply` (working tree only), then `mvn clean deploy -DskipTests -B` using a `~/.m2/settings.xml` `github` server with a classic `write:packages` PAT.
