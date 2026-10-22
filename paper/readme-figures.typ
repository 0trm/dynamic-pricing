// Exports the paper's diagrams, the panel mockup and the badge row as standalone SVGs for the README.
// Build: make figures (or typst compile --root . --input fig=panel --format svg paper/readme-figures.typ assets/panel.svg)

#import "diagrams.typ": context-diagram, architecture-diagram
#import "panel.typ": analyst-panel

#let fig = sys.inputs.at("fig", default: "panel")

// Badges: grey label cell with a grey icon, white value cell, thin border. Text is outlined in the SVG.
#let badges = (
  paper: ("Paper", "PDF"),
  ci: ("CI", "GitHub Actions"),
  python: ("Python", "3.12"),
  models: ("Models", "Prophet + XGBoost"),
  license: ("License", "MIT"),
)
#let badge(name) = {
  let (label, value) = badges.at(name)
  set text(font: ("Helvetica Neue", "Helvetica", "Arial"), size: 8.6pt, top-edge: "cap-height", bottom-edge: "baseline")
  box(stroke: 0.75pt + rgb("#d0d7de"), radius: 3pt, clip: true, grid(
    columns: 2,
    box(fill: rgb("#eaeef2"), inset: (x: 5.5pt, y: 4.5pt), stroke: (right: 0.75pt + rgb("#d0d7de")))[
      #box(baseline: 1.6pt, image("../assets/icons/" + name + ".svg", height: 9pt))#h(3.5pt)#text(fill: rgb("#57606a"), label)
    ],
    box(fill: white, inset: (x: 6pt, y: 4.5pt), text(fill: rgb("#24292f"), value)),
  ))
}

#set page(
  width: auto, height: auto,
  margin: if fig.starts-with("badge-") { 0.5pt } else if fig == "panel" { 1pt } else { 12pt },
  fill: if fig.starts-with("badge-") { none } else { white },
)
#set text(font: "Libertinus Serif", size: 10pt)

#if fig == "context" { scale(140%, reflow: true, context-diagram) }
#if fig == "architecture" { block(width: 7in, architecture-diagram) }
#if fig == "panel" { block(width: 7.2in, analyst-panel(json("figures/numbers.json"))) }
#if fig.starts-with("badge-") { badge(fig.slice(6)) }
