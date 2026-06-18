/* ============================================================
   Langotime ECharts theme.
   Orange-led series palette on a flat, ink-on-paper grid; brand
   fonts (Hanken Grotesk / Space Mono). Registered as "langotime";
   pass it as the 2nd arg:  echarts.init(el, "langotime", {...}).
   Must load AFTER echarts.min.js and BEFORE any echarts.init call.
   ============================================================ */
(function () {
  if (typeof echarts === "undefined" || !echarts.registerTheme) return;

  var ink900 = "#0A0A0A", ink600 = "#444444", ink500 = "#6B6B6B",
      ink300 = "#B8B8B8", ink200 = "#E2E2E2", ink150 = "#ECECEC",
      orange = "#FF4D00";
  var sans = '"Hanken Grotesk", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
  var mono = '"Space Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace';

  var axis = {
    axisLine:  { lineStyle: { color: ink200 } },
    axisTick:  { lineStyle: { color: ink200 } },
    axisLabel: { color: ink500, fontFamily: mono, fontSize: 11 },
    splitLine: { lineStyle: { color: ink150 } },
    splitArea: { show: false }
  };

  echarts.registerTheme("langotime", {
    // orange primary, ink, then telemetry support hues (blue/green/amber/red)
    color: [orange, ink900, "#2D7FF9", "#19A974", "#F2A30F", "#E5484D", "#8A8A8A", "#FF9466"],
    backgroundColor: "transparent",
    textStyle: { fontFamily: sans, color: ink600 },
    title: {
      textStyle: { color: ink900, fontFamily: sans, fontWeight: 800 },
      subtextStyle: { color: ink500, fontFamily: mono }
    },
    line: { itemStyle: { borderWidth: 2 }, lineStyle: { width: 2.5 }, symbolSize: 6, symbol: "circle", smooth: false },
    radar: { name: { textStyle: { color: ink600 } } },
    categoryAxis: axis,
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    legend: { textStyle: { color: ink600, fontFamily: mono } },
    tooltip: {
      backgroundColor: "#FFFFFF", borderColor: ink200, borderWidth: 1,
      textStyle: { color: ink900, fontFamily: sans },
      axisPointer: { lineStyle: { color: ink300 }, crossStyle: { color: ink300 } }
    },
    grid: { borderColor: ink200 },
    visualMap: { textStyle: { color: ink600 } },
    toolbox: { iconStyle: { borderColor: ink500 } }
  });
})();
