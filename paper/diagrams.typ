// System diagrams for the paper, drawn with fletcher so they share the paper's font.

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: pill

#let accent = rgb("#2a78d6")
#let accent-fill = rgb("#eaf2fc")
#let ink-2 = rgb("#6b6a66")
#let line-c = rgb("#4a4a48")

// Engine components: accent; people: pill; external systems: plain.
#let core(pos, body, ..args) = node(pos, body, fill: accent-fill, stroke: 0.7pt + accent, corner-radius: 2pt, ..args)
#let ext(pos, body, ..args) = node(pos, body, fill: white, stroke: 0.6pt + ink-2, corner-radius: 2pt, ..args)
#let person(pos, body, ..args) = node(pos, body, shape: pill, fill: white, stroke: 0.6pt + ink-2, ..args)
#let lbl(body) = text(size: 6.5pt, fill: ink-2, body)
#let small(body) = text(size: 6.5pt, fill: ink-2, body)

#let context-diagram = {
  set text(size: 7.5pt, hyphenate: false)
  set par(justify: false)
  diagram(
    spacing: (22pt, 30pt),
    node-inset: 6pt,
    edge-stroke: 0.6pt + line-c,
    label-size: 6.5pt,
    label-sep: 3pt,
    person((1, 0), align(center)[Pricing analyst #small[(club)]], name: <analyst>, width: 96pt),
    ext((0, 1), align(center)[Club data\ systems\ #small[historical,\ batch]], name: <data>, width: 54pt),
    core((1, 1), align(center)[*Dynamic pricing engine*\ #small[forecast + optimize]], name: <engine>, width: 96pt),
    ext((1, 2), align(center)[Ticketing platform\ #small[live]], name: <ticketing>, width: 96pt),
    person((2, 2), align(center)[Fans], name: <fans>, width: 40pt),

    edge(<data>, <engine>, "-|>", lbl[history], label-side: left),
    edge(<engine>, <analyst>, "-|>", lbl[proposals,\ drift metrics], shift: -16pt, label-side: right),
    edge(<analyst>, <engine>, "-|>", lbl[approvals,\ what-if prices], shift: -16pt, label-side: right),
    edge(<engine>, <ticketing>, "-|>", lbl[approved prices], shift: -16pt, label-side: right),
    edge(<ticketing>, <engine>, "-|>", lbl[sales,\ availability], shift: -16pt, label-side: right),
    edge(<ticketing>, <fans>, "-|>", lbl[tickets], label-side: left),
  )
}

#let architecture-diagram = {
  set text(size: 7pt, hyphenate: false)
  set par(justify: false)
  let enclosure(nodes, paint) = node(
    enclose: nodes, [],
    inset: 11pt, stroke: (paint: paint, thickness: 0.4pt, dash: "dashed"), corner-radius: 3pt, snap: false,
  )
  diagram(
    spacing: (17pt, 10pt),
    node-inset: 5pt,
    edge-stroke: 0.6pt + line-c,
    label-size: 6.5pt,
    label-sep: 2pt,

    ext((0, 0), align(center)[Ticket sales &\ availability], name: <s1>, width: 64pt),
    ext((0, 1), align(center)[Secondary-market\ prices], name: <s2>, width: 64pt),
    ext((0, 2), align(center)[Web & app\ analytics], name: <s3>, width: 64pt),
    ext((0, 3), align(center)[Match calendar\ & channels], name: <s4>, width: 64pt),
    enclosure(((0, 0), (0, 3)), ink-2),
    node((0, -0.95), small[*Data sources*], stroke: none),

    core((1, 1.5), align(center)[Ingestion &\ central store], name: <ingest>, width: 56pt),

    core((2, 0.5), align(center)[Match\ clustering], name: <cluster>, width: 62pt),
    core((2, 2.5), align(center)[Demand\ forecast], name: <forecast>, width: 62pt),
    ext((3, 0.5), align(center)[Business\ constraints], name: <rules>, width: 62pt),
    core((3, 2.5), align(center)[*Decision engine*\ #small[optimize & simulate]], name: <decide>, width: 62pt),
    enclosure((<cluster>, <forecast>, <decide>, <rules>), accent),
    node((2.5, -0.5), text(size: 6.5pt, fill: accent)[*Decision module*], stroke: none),

    person((4.7, 2.5), align(center)[*Analyst panel*\ #small[human in the loop]], name: <panel>, width: 62pt),
    ext((6.3, 2.5), align(center)[Ticketing\ platform], name: <ticket>, width: 52pt),
    ext((6.3, 0.5), align(center)[Fan purchase\ page], name: <fan>, width: 52pt),

    edge(<s1>, <ingest>, "-|>"),
    edge(<s2>, <ingest>, "-|>"),
    edge(<s3>, <ingest>, "-|>"),
    edge(<s4>, <ingest>, "-|>"),
    edge(<ingest>, <cluster>, "-|>"),
    edge(<ingest>, <forecast>, "-|>"),
    edge(<cluster>, <forecast>, "-|>"),
    edge(<forecast>, <decide>, "-|>"),
    edge(<rules>, <decide>, "-|>", lbl[hard rules], label-side: left),
    edge(<decide>, <panel>, "-|>", lbl[proposals,\ warnings], label-side: left, label-pos: 0.6),
    edge(<panel>, <decide>, "-|>", lbl[what-if price], bend: 45deg, label-side: right),
    edge(<panel>, <ticket>, "-|>", lbl[approved\ price (API)], label-side: left, label-pos: 0.5),
    edge(<ticket>, <fan>, "-|>", lbl[new prices,\ price-drop\ alerts], label-side: left),
    edge(<ticket>, (6.3, 4.1), (0, 4.1), <s4>, "-|>", lbl[new sales feed the next forecast], label-pos: 0.5, label-side: left, stroke: (dash: "dashed")),
  )
}
