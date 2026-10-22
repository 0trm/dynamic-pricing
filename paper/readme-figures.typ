// Exports the paper's diagrams and panel mockup as standalone SVGs for the README.
// Build: make figures (or typst compile --root . --input fig=panel --format svg paper/readme-figures.typ assets/panel.svg)

#import "diagrams.typ": context-diagram, architecture-diagram
#import "panel.typ": analyst-panel

#let fig = sys.inputs.at("fig", default: "panel")
#set page(width: auto, height: auto, margin: if fig == "panel" { 1pt } else { 12pt }, fill: white)
#set text(font: "Libertinus Serif", size: 10pt)

#if fig == "context" { scale(140%, reflow: true, context-diagram) }
#if fig == "architecture" { block(width: 7in, architecture-diagram) }
#if fig == "panel" { block(width: 7.2in, analyst-panel(json("figures/numbers.json"))) }
