# Writing a page

For the people filling in `docs/`. Not linked from the nav; it is not part of
the site.

Every page in the site already exists, with the right `<head>`, the right nav
and its own entry marked. **Edit inside `<article class="page">` and leave the
rest of the file alone.** A page you are writing has a `<div class="stub">`
holding a list of what it must cover; delete that block and write the page.

The site is static HTML served by GitHub Pages from `docs/` on the default
branch. There is no generator and no build step. If you find yourself wanting
one, you are doing something the site does not do.

---

## 1. Voice

Plain, present tense. Second person where it helps the reader do something
("you get a table of 42 rows"), third person where it would only be decoration.

- **No marketing.** Nothing is powerful, elegant, seamless, simply, just, or
  easy. If a thing is small, say how small.
- **No history.** Nobody reading this cares what the library used to do, which
  refactor happened, or that something "was previously". Describe what is
  there now. The one exception is a documented refusal — "this is refused
  because …" is present tense and is about the code as it stands.
- **No hedging that carries no information.** "It may be possible to…" means
  either you know or you should go and find out.
- **Say what was measured and what was quoted.** A number you produced by
  running the command is a result. A number you took from
  `documents/real/README.md` is a recorded result, and you say so in the
  sentence that uses it.
- Contractions are fine. Em dashes are fine. Headings are sentence case.
- Keep paragraphs under about six lines. The stylesheet limits prose to a
  34rem measure; a long paragraph is a wall on a phone.

## 2. Every claim is traceable

Beside any claim about how the library behaves, put an HTML comment with the
file and line that says so:

```html
<!-- causalab_mini/plan/spec.py:238 -->
<p>Six write mechanisms, and each one's parameters are checked by name.</p>
```

The comment is for the next person to edit the page, not for the reader. Rules:

- Cite the **code**, not the prose about the code, wherever the code decides
  it. `HANDOFF.md`, `NOTES.md`, `REVIEW.md` and `FINDINGS.md` are citable for
  design intent, for measurements and for history — not for current behaviour.
- Line numbers drift. Cite `path:line` for a specific statement and bare
  `path` for a whole file's job. If you cite a line, quote enough of it in the
  surrounding prose that the next person can find it again if it moved.
- **Do not invent API.** If you cannot find out how something works, read the
  tests — `tests/` has 505 of them and they are the executable spec. If it is
  still unclear, leave the claim out and put a comment where it would have
  gone:

  ```html
  <!-- TODO(author): does a `span` position anchor from the start of the scope
       or the start of the run? tests/test_locate.py does not cover it. -->
  ```

  A TODO in the source is fine. A confident wrong sentence on the page is not.

## 3. Documents are copied, never retyped

Every document a page shows is a real file under `documents/`, copied
**verbatim** — no reformatting, no trimming, no renamed fields — with its path
stated above the block:

```html
<figure class="listing">
  <figcaption><code>documents/v2/das.json</code> <span class="what">— the document this page runs</span></figcaption>
  <pre><code class="language-json">{
  "header": {
…
}</code></pre>
</figure>
```

Escape `&`, `<` and `>` in the content. The `<span class="what">` is optional.

A long document goes in a `<details class="listing">` with the same caption as
its `<summary>`, so the page reads without it and opens for anyone who wants
it:

```html
<details class="listing">
  <summary><code>documents/v2/das.json</code> <span class="what">— the whole document</span></summary>
  <pre><code class="language-json">…</code></pre>
</details>
```

If the document you need does not exist, **do not invent one**. Either use the
closest one that does, or say what is missing in a `TODO(author)` comment.
`documents/v2/` has one document per technique; `documents/real/` has five on
Llama-3.2-1B with recorded results; `documents/*.json` are in the protocol's
own format.

## 4. Commands are run, not imagined

Every command shown on a page was run by the author, and the output pasted is
the output it produced. Include the `$` prompt so the command and its output
are distinguishable:

```html
<pre><code class="language-bash">$ uv run causalab-mini validate documents/v2/das.json
ok: documents/v2/das.json is a valid plan-shaped document</code></pre>
```

- Run from the repository root, with `CUDA_VISIBLE_DEVICES=` set — the CPU
  fixtures expect it and this machine's driver is older than the torch build
  (`CONTRIBUTING.md`). Do not paste the `CUDA_VISIBLE_DEVICES=` prefix into
  the page; it is noise.
- Strip the loader's progress bars and the library warnings. Keep everything
  that is the command's own output.
- **Trim with an explicit `…`** on a line of its own, at the indentation of
  what it replaces. Never silently cut:

  ```
  out/sweep/layers=0,pos=-1/document.json
  …
  out/sweep/layers=15,pos=-1/logit_diff.json
  ```

- Put the date and the machine in an HTML comment above the block:
  `<!-- ran 2026-09-23, CPU, fp32 -->`.
- A timing you quote is a timing you measured. `documents/real/README.md`
  records wall times on an A100 and through NDIF; quote those as recorded.

A `documents/v2/` document runs on a tiny random-weight Llama in a few seconds
on CPU, so there is no excuse for not running it. A `documents/real/` document
needs Llama-3.2-1B (gated: `huggingface-cli login`); it is about 2.4 GB and the
layer sweep takes 93 seconds on CPU.

## 5. Code blocks

| what | markup |
|---|---|
| JSON | `<pre><code class="language-json">` |
| a shell session | `<pre><code class="language-bash">` |
| Python | `<pre><code class="language-python">` |
| a directory tree, a plain listing | `<pre><code>` |

There is no syntax highlighter on the site. The `language-*` class is there so
one can be added later and so the markup says what the block is. Inline code is
`<code>` with no class.

## 6. The page template

```html
<header>
  <p class="eyebrow">Reference</p>              <!-- or: Use case -->
  <h1>Positions</h1>
  <p class="lede">One sentence saying what the page is for.</p>
</header>

<p>The lede paragraph: what this is, in prose, in about four to six lines. A
reader who stops here should have the idea.</p>

<h2>At a glance</h2>
<!-- a real snippet: the smallest document, command or output that shows the
     thing working. Never a sketch, never a "for example, you might write". -->

<h2>…sections…</h2>

<div class="seealso">
  <h2>See also</h2>
  <ul>
    <li><a href="…">Page</a> — why you would go there next</li>
  </ul>
</div>
```

Order within a page: what it is → the smallest thing that works → the field or
option reference (a table) → the cases that are not obvious → what is refused
and why → see also. A "see also" entry always says *why*, never just names the
page.

### The pieces the stylesheet gives you

| markup | what it is for |
|---|---|
| `<p class="lede">` | the one-sentence subtitle under `<h1>` |
| `<p class="eyebrow">` | the small label above `<h1>` |
| `<figure class="listing">` + `<figcaption>` | a code block with its source stated |
| `<details class="listing">` + `<summary>` | the same, folded away |
| `<div class="note"><span class="label">…</span>` | an aside worth stopping for |
| `<div class="caution"><span class="label">…</span>` | a trap, or what a result does **not** say |
| `<table>` | a field reference. Use `class="num"` on numeric cells |
| `<ul class="cards">` with `<span class="title">` / `<span class="blurb">` | a directory of pages |
| `<div class="bars">` | a bar chart made of divs — see below |
| `<div class="seealso">` | the footer |

A bar chart for something with one number per label — heads, layers, features
— without drawing an SVG. The width of each `.fill` is the only thing you set,
as a percentage:

```html
<div class="bars">
  <div class="row head"><span class="key">head</span><span>logit difference</span><span class="val"></span></div>
  <div class="row"><span class="key">L12 H28</span>
    <span class="track"><span class="fill" style="width:100%"></span></span>
    <span class="val">+0.05</span></div>
  <div class="row"><span class="key">L12 H11</span>
    <span class="track"><span class="fill" style="width:12%"></span></span>
    <span class="val">-2.11</span></div>
</div>
```

If the numbers cross zero, say in the caption what the bar is measuring from.
A bar chart of sixteen rows is fine; of ninety-six, use a table or an SVG.

Use a table for anything with more than three parallel entries — components,
mechanisms, flags, metric kinds. Prose that lists six things is a table that
has not been written yet.

## 7. Diagrams

Draw one only when it shows a **mechanism** the prose cannot: something moving,
something looping back, two things being the same object. A picture of a list
is a list.

Inline SVG, in the page, no image files. It must read in both themes, which
means: **no literal colours.** Use the stylesheet's tokens.

```html
<figure>
  <svg viewBox="0 0 640 200" role="img" aria-labelledby="d-t d-d" width="640">
    <title id="d-t">Short name of the diagram</title>
    <desc id="d-d">What it shows, in a sentence, for a reader who cannot see it.</desc>
    <rect x="14" y="24" width="150" height="52" rx="6"
          fill="var(--bg-raised)" stroke="var(--rule-strong)"/>
    <text x="89" y="48" text-anchor="middle" font-size="14" font-weight="600"
          fill="var(--fg)" font-family="ui-sans-serif, system-ui, sans-serif">a box</text>
    <path d="M 168 50 L 240 50" stroke="var(--fg-faint)" stroke-width="1.4"
          fill="none" marker-end="url(#arrow)"/>
  </svg>
  <figcaption>What to take from it.</figcaption>
</figure>
```

- Palette: `--fg`, `--fg-muted`, `--fg-faint` for text and lines;
  `--bg-raised` for a box fill; `--rule` / `--rule-strong` for borders;
  `--accent` for the one thing being emphasised. One accent per diagram.
- `viewBox` plus `width`, no `height` — the stylesheet makes it fluid.
- Every `id` in an SVG is global to the page. Prefix them (`pipe-t`,
  `curve-t`) so two diagrams on one page do not collide; that includes
  `<marker id="arrow">`, which needs its own `<defs>` per diagram or a
  page-unique id.
- `role="img"` with `<title>` and `<desc>` is not optional.
- Font sizes below 10 are unreadable on a phone. Below 62rem the stylesheet
  stops shrinking a diagram at `min-width: 490px` and lets the figure scroll
  sideways, so a label at `font-size="10.5"` lands at about 8px there. A
  narrower `viewBox` avoids the sideways scroll entirely: 460 fits a phone's
  column at 0.78x, where 640 would be 0.51x. Draw narrow if you can.

`docs/index.html` and `docs/use-cases/layer-sweep.html` each have one; copy
their shape.

## 8. The nav

Every page carries the same nav markup. It is already in your page — **do not
re-paste it**, and if you touch it, touch only the `aria-current` attribute.

Two things to know:

1. **Hrefs are relative to the page.** A page at `docs/` uses
   `href="documents.html"`; a page at `docs/use-cases/` uses
   `href="../documents.html"`. The snippet below is the top-level form; inside
   `use-cases/`, every `href` gains a `../` and so do the `site.css` and
   `site.js` links in `<head>` and at the end of `<body>`.
2. **Your page marks its own entry**, in the markup, with
   `aria-current="page"` — exactly one per page. `site.js` does not do this,
   deliberately: the highlight is right before a byte of JavaScript has run.

The snippet, for a page at the top level. Everything from `<a class="skip">`
down to `<main id="content">`:

```html
<a class="skip" href="#content">Skip to content</a>

<div class="topbar">
  <button class="iconbutton" data-nav-toggle aria-expanded="false" aria-controls="sidebar">menu</button>
  <span class="brand">causalab-mini</span>
  <button class="iconbutton js-only" data-theme-toggle style="margin-left:auto">dark</button>
</div>

<div class="layout">
  <aside class="sidebar" id="sidebar">
    <a class="brand" href="index.html">causalab-mini</a>
    <span class="brand-sub">a plan is pure data; an engine turns it into tensors</span>

    <nav aria-label="Documentation">
      <ol>
        <li><a href="index.html">Home</a></li>
      </ol>

      <h2>The document</h2>
      <ol>
        <li><a href="documents.html">Documents</a></li>
        <li><a href="locations.html">Locations</a></li>
        <li><a href="positions.html">Positions</a></li>
        <li><a href="ops.html">Reads, writes, metrics</a></li>
      </ol>

      <h2>Use cases</h2>
      <ol>
        <li><a href="use-cases/index.html">All use cases</a></li>
      </ol>
      <ol class="sub">
        <li><a href="use-cases/layer-sweep.html">Layer sweep</a></li>
        <li><a href="use-cases/entity-patching.html">Entity patching</a></li>
        <li><a href="use-cases/das-fit.html">DAS fit</a></li>
        <li><a href="use-cases/dbm-fit.html">DBM fit</a></li>
        <li><a href="use-cases/attention-pattern.html">Attention pattern</a></li>
        <li><a href="use-cases/logit-lens.html">Logit lens</a></li>
        <li><a href="use-cases/generation.html">Generation</a></li>
        <li><a href="use-cases/sae-features.html">SAE features</a></li>
        <li><a href="use-cases/batching-and-remote.html">Batching &amp; remote</a></li>
        <li><a href="use-cases/sweeps.html">Sweeps</a></li>
      </ol>

      <h2>Running</h2>
      <ol>
        <li><a href="cli.html">The CLI</a></li>
      </ol>

      <h2>Understand</h2>
      <ol>
        <li><a href="engine.html">The engine</a></li>
        <li><a href="standardization.html">Standardization</a></li>
        <li><a href="remote.html">Remote &amp; NDIF</a></li>
      </ol>
    </nav>

    <div class="footer">
      <p><a href="https://github.com/JadenFiotto-Kaufman/causalab-mini">source on GitHub</a></p>
      <p><button class="iconbutton js-only" data-theme-toggle>dark</button></p>
    </div>
  </aside>

  <main id="content">
```

If you add a page to the site, it goes in this snippet **and** in every page
that already exists. That is the cost of having no generator, and it is why the
structure is fixed.

## 9. `<head>` and `<title>`

```html
<title>Positions — causalab-mini</title>
<meta name="description" content="One sentence. It is what a search result shows.">
```

The title is the nav label plus ` — causalab-mini`. Do not change your page's
title to something more descriptive; the nav label and the title agree on
purpose.

## 10. Before you say a page is done

```bash
cd docs && python -m http.server 8000     # then open http://localhost:8000/
```

If you are on a box with no desktop, Firefox renders headless and writes a
PNG, which is enough to catch an overflowing table or a diagram whose labels
have gone under 6px:

```bash
firefox --headless --window-size=1200,2400 --screenshot /tmp/wide.png  http://localhost:8000/ops.html
firefox --headless --window-size=360,3000  --screenshot /tmp/phone.png http://localhost:8000/ops.html
```

For the dark theme, a headless render follows `prefers-color-scheme`, so the
quickest way to see the other one is a wrapper page that frames yours inside
an `<html data-theme="dark">` — the same attribute `site.js` sets from the
toggle. Do not commit the wrapper.

- Read the page in both themes. The toggle is in the sidebar footer.
- Read it at 360px wide. No horizontal scrolling of the page itself; a wide
  code block or table scrolls inside its own box, which is what the stylesheet
  does already.
- Every link on the page resolves, including the "see also" ones — the pages
  you link to exist already, even the ones that are still stubs.
- Every code block you added was produced by a command you ran.
- Exactly one `aria-current="page"` in the file.
- No `TODO(author)` left that you could have answered by reading the code.
- Every `<h2>` and `<h3>` you added carries an `id` and the `#` anchor link:
  `<h2 id="what-a-read-is">What a read is <a class="anchor" href="#what-a-read-is" aria-label="Link to this section">#</a></h2>`.
  The id is the heading's text, lowercased, with runs of punctuation and
  spaces turned into single hyphens; it has to be unique in the file.
