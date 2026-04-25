import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const MARGIN = { top: 16, right: 24, bottom: 32, left: 96 };
const MIN_WIDTH = 220;
const MIN_HEIGHT = 240;

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
  const [size, setSize] = useState({ width: 0, height: 0 });

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
      setSize({
        width: Math.max(MIN_WIDTH, Math.floor(rect.width)),
        height: Math.max(MIN_HEIGHT, Math.floor(rect.height)),
      });
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
    if (size.width === 0 || size.height === 0) return;

    const { width: outerW, height: outerH } = size;
    const width = outerW - MARGIN.left - MARGIN.right;
    const height = outerH - MARGIN.top - MARGIN.bottom;

    d3.select(svgEl)
      .attr("viewBox", `0 0 ${outerW} ${outerH}`)
      .attr("preserveAspectRatio", "none")
      .attr("width", outerW)
      .attr("height", outerH);

    const svg = d3
      .select(svgEl)
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`)
      .attr("fill", "white");

    const xMax = d3.max(data, (d) => d.prob) || 1;
    const xScale = d3.scaleLinear().domain([0, xMax]).range([0, width]);
    const yScale = d3
      .scaleBand()
      .domain(data.map((d) => d.text))
      .range([0, height])
      .padding(0.1);

    svg
      .append("g")
      .attr("transform", `translate(0, ${height})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll("text")
      .attr("fill", "none");

    const yAxis = svg.append("g").call(d3.axisLeft(yScale));
    yAxis.selectAll("text").attr("fill", "#f5f6fa");

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
        const tickText = d3.select(this).select("text").text();
        if (onTickClickRef.current) onTickClickRef.current(tickText);
      });

    svg
      .selectAll(".bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("y", (d) => yScale(d.text))
      .attr("x", 0)
      .attr("height", yScale.bandwidth())
      .attr("width", (d) => xScale(d.prob))
      .attr("fill", "white");

    svg
      .selectAll(".label")
      .data(data)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", (d) => xScale(d.prob) - 5)
      .attr("y", (d) => yScale(d.text) + yScale.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("dx", "8px")
      .text((d) => (Number(d.prob) === 0 ? "<0.01" : d.prob))
      .attr("fill", "white");
  }, [data, size]);

  const hasData = data && data.length > 0;

  return (
    <div ref={containerRef} className="w-full h-full relative">
      {hasData ? (
        <svg ref={svgRef} className="barplot-svg block" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-center text-fg/50 text-xs px-4">
          Tap any generated token to inspect the top-k candidate probabilities
          here.
        </div>
      )}
    </div>
  );
}
