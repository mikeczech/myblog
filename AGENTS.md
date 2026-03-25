# Repository Guidelines

## Project Structure & Module Organization
This repository is a Hugo site for `czechthedata.com`. Content lives in `content/posts/` and standalone pages such as `content/impressum.md`. Site configuration is in `hugo.yaml`. Custom templates and overrides live in `layouts/` and `assets/`, while published images and static files live in `static/`. The theme is vendored as the `themes/PaperMod` git submodule; prefer overriding it from `layouts/` or `assets/` instead of editing theme files directly.

## Build, Test, and Development Commands
Use Hugo for all local development:

- `git submodule update --init --recursive`: fetch the PaperMod theme after clone.
- `hugo server -D`: run the site locally and include draft posts.
- `hugo --gc --minify`: produce a production-style build in `public/`.
- `ruby mdtoc.rb content/posts/<post>.md`: generate a Markdown table of contents for long posts.

GitHub Pages deploys from `.github/workflows/hugo.yaml` using Hugo `0.151.2`.

## Coding Style & Naming Conventions
Keep Markdown content concise and front matter consistent with existing posts: TOML front matter delimited by `+++`, e.g. `title`, `date`, `draft`, and `tags`. Post filenames use lowercase snake_case, such as `content/posts/context_engineering.md`. Preserve existing indentation in the file you touch: YAML uses spaces, Hugo templates/CSS should stay readable and minimal. Prefer adding site-specific overrides under `layouts/partials/` or `assets/css/` rather than patching upstream theme code.

## Testing Guidelines
There is no automated test suite in this repo. Validate changes by running `hugo server -D` for local review and `hugo --gc --minify` before opening a PR. For content edits, check front matter, internal links, code fences, math rendering, and image paths under `static/`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Add qdrant article` and `Update hugo`. Follow that pattern: one clear action per commit, ideally under 50 characters. PRs should include a brief summary, note any config or deployment impact, and attach screenshots for visual changes affecting layouts, CSS, or rendered posts. Link the related issue when applicable.
