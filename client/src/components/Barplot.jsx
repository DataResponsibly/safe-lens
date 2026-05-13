import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const MARGIN = { top: 16, right: 48, bottom: 8, left: 160 };
const MIN_WIDTH = 220;
const BAR_HEIGHT = 18;
const BAR_GAP = 8;
const ROW_HEIGHT = BAR_HEIGHT + BAR_GAP;
const PADDING_INNER = BAR_GAP / ROW_HEIGHT;
const TICK_FONT_SIZE = 15;
const LABEL_FONT_SIZE = 14;

/**
 * React wrapper around the original D3 barplot from client/js/main.js (barplot_new).
 * Renders a horizontal bar chart of candidate tokens with their probabilities.
 * Observes its container so the chart always fills the available space (no
 * fixed-aspect letterboxing). Clicking a tick label fires
 * `onTickClick(tokenText)` so the parent can trigger a regenerate request.
 */
export default function Barplot({ data, onTickClick, disabled }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const onTickClickRef = useRef(onTickClick);
  const disabledRef = useRef(disabled);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    onTickClickRef.current = onTickClick;
  }, [onTickClick]);
  useEffect(() => {
    disabledRef.current = disabled;
  }, [disabled]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      setWidth(Math.max(MIN_WIDTH, Math.floor(rect.width)));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    d3.select(svgEl).selectAll("*").remove();

    if (!data || data.length === 0) return;
    if (width === 0) return;

    const plotData = data.slice(0, 50);
    const n = plotData.length;

    const outerW = width;
    const contentHeight = n * ROW_HEIGHT - BAR_GAP;
    const outerH = MARGIN.top + contentHeight + MARGIN.bottom;
    const innerWidth = outerW - MARGIN.left - MARGIN.right;

    d3.select(svgEl)
      .attr("viewBox", `0 0 ${outerW} ${outerH}`)
      .attr("width", outerW)
      .attr("height", outerH);

    const svg = d3
      .select(svgEl)
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`)
      .attr("fill", "white");

    const xMax = d3.max(plotData, (d) => d.prob) || 1;
    const xScale = d3.scaleLinear().domain([0, xMax]).range([0, innerWidth]);
    const yScale = d3
      .scaleBand()
      .domain(plotData.map((d) => d.text))
      .range([0, contentHeight])
      .paddingInner(PADDING_INNER)
      .paddingOuter(0);

    const yAxis = svg.append("g").call(d3.axisLeft(yScale));
    yAxis
      .selectAll("text")
      .style("font-size", `${TICK_FONT_SIZE}px`)
      .style("font-weight", "500");

    yAxis
      .selectAll("text")
      .text((d) => {
        if (d === " ") return "[sp]";
        if (d === "\n") return "↵";
        if (d === "\t") return "→";
        if (typeof d === "string" && d.trim() === "" && d.length > 0)
          return `[${d.length}sp]`;
        return d;
      });

    yAxis
      .selectAll(".tick")
      .style("cursor", "pointer")
      .on("click", function () {
        if (disabledRef.current) {
          alert(
            "Token editing is not allowed while SafeNudge is activated."
          );
          return;
        }
        const token = d3.select(this).datum();
        if (onTickClickRef.current) onTickClickRef.current(token);
      });

    svg
      .selectAll(".bar")
      .data(plotData)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("y", (d) => yScale(d.text))
      .attr("x", 0)
      .attr("height", yScale.bandwidth())
      .attr("width", (d) => xScale(d.prob));

    svg
      .selectAll(".label")
      .data(plotData)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", (d) => xScale(d.prob) - 5)
      .attr("y", (d) => yScale(d.text) + yScale.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("dx", "8px")
      .style("font-size", `${LABEL_FONT_SIZE}px`)
      .text((d) => (Number(d.prob) === 0 ? "<0.01" : d.prob));
  }, [data, width]);

  const hasData = data && data.length > 0;

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-y-auto">
      {hasData ? (
        <svg ref={svgRef} className="barplot-svg block" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-center text-fg text-xs px-4">
          Tap any generated token to inspect the top-k candidate probabilities
          here.
        </div>
      )}
    </div>
  );
}
