# docs/

The causalab-mini site, served by GitHub Pages from this folder. It is static
HTML: no generator, no build step, no framework. `index.html` is the landing
page; the four reference pages (`documents`, `locations`, `positions`, `ops`)
describe the document format; `use-cases/` holds one page per worked
experiment, each built on a real document from `documents/`; `cli.html` is the
command line; and `engine.html`, `standardization.html` and `remote.html`
explain how a plan becomes tensors. Every page shares `site.css` and the small
`site.js`, which does two things and nothing else — the light/dark override and
the nav toggle at phone width. Each page carries its own copy of the nav markup
and marks its own entry with `aria-current="page"`, so adding a page means
editing every page; that is the price of having no build step, and the reason
the structure is fixed. `STYLE.md` is the guide for anyone writing a page and
is not part of the site.

To preview it, serve this directory and open it in a browser — opening the
files directly with `file://` works too, but a server is closer to what GitHub
Pages does:

```bash
cd docs
python -m http.server 8000     # http://localhost:8000/
```

Nothing is fetched from a CDN, so the site works offline. Pages under
`use-cases/` link back with `../`; everything else is flat.
