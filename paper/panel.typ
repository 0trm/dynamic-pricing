// Mockup of the analyst control panel. Every value comes from the evaluation model
// via figures/numbers.json (written by figures.py); nothing here is typed by hand.

#import "@preview/cetz:0.3.4"

#let ui-font = ("Helvetica Neue", "Helvetica", "Arial")
#let accent = rgb("#2a78d6")
#let accent-soft = rgb("#eaf2fc")
#let warn = rgb("#b45309")
#let warn-soft = rgb("#fdf3e3")
#let ink = rgb("#1f2328")
#let ink-2 = rgb("#5f6670")
#let line-c = rgb("#d9dde3")
#let surface = rgb("#f6f7f9")

#let thousands(n) = {
  let s = str(calc.round(n))
  let out = ""
  for (i, ch) in s.clusters().rev().enumerate() {
    if i > 0 and calc.rem(i, 3) == 0 { out = "," + out }
    out = ch + out
  }
  out
}
#let eur(n) = "€" + thousands(n)
#let pct(x) = {
  let v = calc.round(x * 100, digits: 1)
  let a = str(calc.abs(v))
  (if v > 0 { "+" } else if v < 0 { "−" } else { "" }) + (if a.contains(".") { a } else { a + ".0" }) + "%"
}

#let pill(body, fg, bg) = box(fill: bg, inset: (x: 4pt, y: 2pt), radius: 6pt, text(fill: fg, size: 5.5pt, weight: "medium", body))
#let kpi(label, value, ..extra, highlight: false) = {
  let note = extra.pos().at(0, default: none)
  box(
    width: 100%, height: 34pt, inset: 5pt, radius: 3pt,
    fill: if highlight { accent-soft } else { white },
    stroke: 0.5pt + (if highlight { accent } else { line-c }),
  )[
    #text(size: 5.5pt, fill: ink-2, label)\
    #text(size: 10pt, weight: "bold", fill: if highlight { accent } else { ink }, value)
    #if note != none [\ #text(size: 5.5pt, fill: ink-2, note)]
  ]
}
#let button(body, primary: false) = box(
  inset: (x: 7pt, y: 4pt), radius: 3pt,
  fill: if primary { accent } else { white },
  stroke: 0.6pt + (if primary { accent } else { line-c }),
  text(size: 6.5pt, weight: "medium", fill: if primary { white } else { ink }, body),
)

#let revenue-chart(q, width: 9cm, height: 3.2cm) = {
  let pts = q.curve
  let (lo, hi) = q.band
  let ymax = calc.max(..pts.map(p => p.at(1))) * 1.12
  let x(v) = (v - lo) / (hi - lo) * width / 1cm
  let y(v) = v / ymax * height / 1cm
  let near(price) = pts.map(p => p.at(1)).at(pts.map(p => calc.abs(p.at(0) - price)).position(d => d == calc.min(..pts.map(p => calc.abs(p.at(0) - price)))))
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let ystep = if ymax > 4000 { 2000 } else { 500 }
    for t in range(0, int(ymax), step: ystep) {
      line((0, y(t)), (x(hi), y(t)), stroke: 0.3pt + line-c)
      content((-0.08, y(t)), anchor: "east", text(size: 5pt, fill: ink-2, eur(t)))
    }
    for t in range(int(calc.ceil(lo / 20) * 20), hi + 1, step: 20) {
      content((x(t), -0.12), anchor: "north", text(size: 5pt, fill: ink-2, eur(t)))
    }
    line((0, 0), (x(hi), 0), stroke: 0.5pt + ink-2)
    line(..pts.map(p => (x(p.at(0)), y(p.at(1)))), stroke: 1.1pt + accent)
    // current price (projected revenue at the current price) and recommendation
    let cur = q.current_price
    line((x(cur), 0), (x(cur), y(q.current_revenue)), stroke: (paint: ink-2, thickness: 0.5pt, dash: "dashed"))
    content((x(cur) - 0.06, 0.1), anchor: "south-east", text(size: 5pt, fill: ink-2)[current #eur(cur)])
    circle((x(cur), y(q.current_revenue)), radius: 0.06, fill: white, stroke: 0.8pt + ink-2)
    circle((x(q.recommended_price), y(q.projected_revenue)), radius: 0.075, fill: accent, stroke: 0.8pt + white)
    content((x(q.recommended_price) - 0.12, y(q.projected_revenue) + 0.08), anchor: "south-east", text(size: 5.5pt, fill: accent, weight: "bold")[recommended #eur(q.recommended_price)])
  })
}

#let analyst-panel(data) = {
  set text(font: ui-font, size: 6.5pt, fill: ink, hyphenate: false)
  set par(justify: false, leading: 0.45em, first-line-indent: 0pt, spacing: 0.5em)
  set align(left)
  show table: set text(size: 6.5pt)
  let p = data.panel
  let q = p.queue.find(r => r.zone == p.selected_zone)
  let pts = q.curve
  let whatif = q.recommended_price - 15
  let whatif-pt = pts.find(r => r.at(0) == whatif)
  let flagged = p.queue.filter(r => r.at_upper_edge or r.current_price > r.band.at(1))
  let acc = data.accuracy.ensemble

  block(width: 100%, radius: 5pt, stroke: 0.6pt + line-c, clip: true, fill: white)[
    // App bar
    #block(width: 100%, fill: ink, inset: (x: 9pt, y: 6pt), below: 0pt)[
      #grid(columns: (auto, 1fr, auto), align: horizon, column-gutter: 14pt,
        text(fill: white, weight: "bold", size: 7.5pt)[Pricing console],
        {
          set text(size: 6.5pt)
          text(fill: white)[Proposals]; h(12pt); text(fill: rgb("#9aa4b1"))[Simulation]; h(12pt); text(fill: rgb("#9aa4b1"))[Model health]; h(12pt); text(fill: rgb("#9aa4b1"))[History]
        },
        box(fill: rgb("#3a4552"), radius: 6pt, inset: (x: 5pt, y: 2pt), text(fill: white, size: 6pt)[Pricing analyst]),
      )
    ]
    // Context bar
    #block(width: 100%, fill: surface, inset: (x: 9pt, y: 5pt), below: 0pt, stroke: (bottom: 0.5pt + line-c))[
      #grid(columns: (1fr, auto), align: horizon,
        [*Match #p.match_id* · opponent tier #p.opponent_tier · *#p.days_until_match days to kick-off* · weather: #lower(p.weather)],
        text(fill: ink-2)[Model: holdout WAPE #calc.round(acc.wape * 100, digits: 1)% · elasticity check passed · price band = observed p5–p95],
      )
    ]
    #block(width: 100%, inset: 9pt)[
      #grid(columns: (1.55fr, 1fr), column-gutter: 12pt,
        // Left: selected proposal
        [
          #text(size: 8pt, weight: "bold")[#q.zone] #h(4pt) #pill([Pending review], accent, accent-soft)
          #v(2pt)
          #grid(columns: (1fr,) * 4, column-gutter: 5pt,
            kpi[Current price][#eur(q.current_price)],
            kpi(highlight: true)[Recommended price][#eur(q.recommended_price)][#pct(q.recommended_price / q.current_price - 1) vs current],
            kpi[Predicted sales][#str(calc.round(q.predicted_sales)) / day][tickets at recommended price],
            kpi[Projected revenue][#eur(q.projected_revenue)][#pct(q.projected_revenue / q.current_revenue - 1) vs current price],
          )
          #v(3pt)
          #text(size: 6pt, weight: "bold")[Projected daily revenue by price]
          #text(size: 5.5pt, fill: ink-2)[ (searched within the observed band #eur(q.band.at(0))–#eur(q.band.at(1)))]
          #v(1pt)
          #align(center, revenue-chart(q))
          #v(3pt)
          #grid(columns: (auto, 1fr, auto), align: horizon, column-gutter: 6pt,
            text(size: 6pt, weight: "bold")[What-if],
            {
              let frac = (whatif - q.band.at(0)) / (q.band.at(1) - q.band.at(0))
              box(width: 100%, height: 8pt, {
                place(horizon, line(length: 100%, stroke: 1.5pt + line-c))
                place(horizon, line(length: frac * 100%, stroke: 1.5pt + accent))
                place(horizon, dx: frac * 100% - 3.5pt, circle(radius: 3.5pt, fill: white, stroke: 1pt + accent))
              })
            },
            text(size: 6pt)[#eur(whatif) → #str(calc.round(whatif-pt.at(1) / whatif)) tickets · #eur(whatif-pt.at(1))],
          )
          #v(4pt)
          #button(primary: true)[Approve #eur(q.recommended_price)] #h(3pt) #button[Edit price] #h(3pt) #button[Reject]
        ],
        // Right: queue for this match
        [
          #text(size: 8pt, weight: "bold")[Proposal queue] #h(3pt) #text(fill: ink-2)[match #p.match_id, today]
          #v(2pt)
          #table(
            columns: (1fr, auto, auto, auto),
            align: (left, right, right, right),
            stroke: (x, y) => if y > 0 { (top: 0.4pt + line-c) },
            inset: (x: 3pt, y: 3.6pt),
            fill: (x, y) => if y > 0 and p.queue.at(y - 1).zone == p.selected_zone { accent-soft },
            table.header(..([Zone], [Current], [Proposed], [Δ revenue]).map(h => text(size: 5.5pt, fill: ink-2, weight: "medium", h))),
            ..p.queue.map(r => (
              [#r.zone #if r.at_upper_edge or r.current_price > r.band.at(1) { h(2pt); pill([check], warn, warn-soft) }],
              eur(r.current_price),
              text(weight: "bold")[#eur(r.recommended_price)],
              text(fill: if r.projected_revenue >= r.current_revenue { rgb("#18794e") } else { warn }, pct(r.projected_revenue / r.current_revenue - 1)),
            )).flatten(),
          )
          #v(3pt)
          #for r in flagged {
            block(width: 100%, fill: warn-soft, radius: 3pt, inset: 5pt, stroke: 0.5pt + rgb("#f0c987"))[
              #text(fill: warn, weight: "bold")[Warning · #r.zone] \
              #text(size: 6pt)[Current price #eur(r.current_price) is above the observed range (#eur(r.band.at(0))–#eur(r.band.at(1))); the proposal sits at the band's upper edge. Review before approving.]
            ]
          }
          #v(2pt)
          #text(size: 5.5pt, fill: ink-2)[Approved prices are pushed to the ticketing platform through the API.]
        ],
      )
    ]
  ]
}
