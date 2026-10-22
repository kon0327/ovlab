"use strict";
(() => {
  const data = JSON.parse(document.getElementById("dashboard-data").textContent);
  const $ = id => document.getElementById(id);
  const node = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  const fmt = value => value == null ? "Unavailable" : typeof value === "number" ? value.toLocaleString(undefined, {maximumFractionDigits: 5}) : String(value);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const card = (title, value, caption) => { const e = node("div", null, "card"); e.append(node("div", title, "caption"), node("div", fmt(value), "kpi")); if (caption) e.append(node("p", caption, "caption")); return e; };
  Object.entries(data.counts).forEach(([key, value]) => $("kpis").append(card(key.replaceAll("_", " "), value)));
  data.notes.forEach(text => $("notes").append(node("p", text)));
  const train = data.mode === "train";
  const settings = data.hyperparameters.training || {};
  const flowCard = (title, body) => {const e = node("div", null, "card"); e.append(node("h3", title), node("p", body)); return e;};
  const evidence = data.evidence;
  const chunkTrain = train && evidence?.objective === "masked_l1";
  $("flow-content").append(
    flowCard("Input", train ? `${fmt(data.counts.dataset_records)} dataset records → ${fmt(data.counts.examples_seen)} examples seen. Image tensors + prompt tokens + attention mask + ${chunkTrain ? "continuous future actions and a validity mask" : "supervised action-token labels"}.` : `${fmt(data.counts.observations)} observations: camera images, instruction, proprioception. ${fmt(data.counts.predictions)} policy calls.`),
    node("div", "→", "arrow"),
    flowCard("Model", train ? `Microbatch: ${fmt(settings.per_device_batch_size)} records. Accumulation: ${fmt(settings.gradient_accumulation_steps)} microbatches per optimizer step. Precision: ${fmt(settings.precision)}.` : "One observation per canonical policy call. Internal tokenization and tensor batching are available only if captured in policy metadata."),
    node("div", "→", "arrow"),
    flowCard("Output", train ? `${fmt(data.counts.optimizer_steps)} optimizer steps → ${chunkTrain ? "continuous action chunks and masked L1 loss" : "next-token logits and loss"} → checkpoint. See the captured batch and first record below.` : `${fmt(data.counts.predictions)} prediction chunks → ${fmt(data.counts.predicted_actions)} predicted actions → ${fmt(data.counts.executed_actions)} applied actions. Predicted and applied commands remain separate.`)
  );
  const detail = (title, value) => {
    const e = node("div", null, "card"); e.append(node("h3", title));
    if (value == null) {e.append(node("p", "Unavailable — this evidence was not recorded.")); return e;}
    const entries = typeof value === "object" && !Array.isArray(value) ? Object.entries(value) : [["Values", value]];
    entries.forEach(([key, item]) => {
      if (key === "images") {
        const images = node("div", null, "image-row");
        item.forEach(image => {const figure = node("figure"); if (image.preview) {const img = node("img"); img.src = image.preview; img.alt = image.name + " canonical RGB preview"; figure.append(img);} figure.append(node("figcaption", `${image.name} · ${image.shape.join(" × ")} · ${image.dtype}`)); if (!image.preview) figure.append(node("p", "RGB preview not embedded for this record; array reference is retained.", "caption")); images.append(figure);});
        e.append(images);
      }
      if (key === "processed_image_preview" && item?.values?.length) {
        const pixels=item.values, h=pixels.length, w=pixels[0].length, canvas=node("canvas"), values=pixels.flat(2).filter(finite);
        const lo=Math.min(...values), hi=Math.max(...values), span=hi-lo||1;
        canvas.width=w;canvas.height=h;canvas.style.width="192px";canvas.style.imageRendering="pixelated";
        const ctx=canvas.getContext("2d"), image=ctx.createImageData(w,h);
        for(let y=0;y<h;y++)for(let x=0;x<w;x++){for(let c=0;c<3;c++)image.data[(y*w+x)*4+c]=Math.round(((pixels[y][x][c]??lo)-lo)/span*255);image.data[(y*w+x)*4+3]=255;}ctx.putImageData(image,0,0);
        e.append(canvas,node("p",item.semantics,"caption"));
      }
      const d = node("details"); const summary = node("summary", key.replaceAll("_", " ")); d.append(summary);
      if (typeof item === "object" && item != null) d.append(node("pre", JSON.stringify(item, (k,v) => k === "preview" ? "[embedded image]" : v, 2))); else d.append(node("p", fmt(item)));
      if (["instruction", "actions", "executed", "shape", "input_ids", "labels", "predicted_tokens", "loss"].includes(key)) d.open = true;
      e.append(d);
    });
    return e;
  };
  if (train) {
    $("batch").append(node("p", evidence ? `Actual captured microbatch ${evidence.microbatch_index}; ${fmt(evidence.batch_size)} records. Tensor shapes describe the full batch. First-record previews are bounded; total element counts and truncation are explicit.` : "No batch capture exists in this historical run. Hyperparameters show the configured batch size; actual examples are unavailable."));
    $("record").append(detail("Input batch", evidence?.input_batch), detail("Input record · index 0", evidence?.input_record), detail("Output batch", evidence?.output_batch), detail("Output record · index 0", evidence?.output_record));
  } else if (data.tasks.length) {
    const taskSelect = node("select"), episodeSelect = node("select"), recordSlider = node("input");
    recordSlider.type = "range"; recordSlider.min = "0"; recordSlider.step = "1";
    const recordLabel = node("label", "Prediction / policy call"); recordLabel.append(recordSlider);
    for (const [label, control] of [["Task", taskSelect], ["Episode", episodeSelect]]) {const l = node("label", label); l.append(control); $("selectors").append(l);}
    $("selectors").append(recordLabel);
    data.tasks.forEach((task, i) => {const option = node("option", task.task_id); option.value = i; taskSelect.append(option);});
    const renderRecord = () => {
      const task = data.tasks[Number(taskSelect.value)], episode = task.episodes[Number(episodeSelect.value)];
      $("record").replaceChildren(); $("batch").replaceChildren();
      if (!episode) {$("batch").append(node("p", "No finalized episodes in this task.")); return;}
      const row = episode.records[Number(recordSlider.value)];
      $("batch").append(node("p", `Rollout ${episode.rollout_index} · seed ${episode.seed} · ${episode.terminal_status} · ${episode.counts.predictions} calls · ${episode.counts.executed_actions} executed actions. ${episode.records.length < episode.counts.predictions ? "The run overview embeds the first three calls; open the task page in the navigation to inspect every call." : "Every recorded prediction is selectable."}`));
      if (!row) {$("record").append(detail("Policy input", null), detail("Policy output", null)); return;}
      recordLabel.firstChild.textContent = `Prediction ${row.index + 1} / ${episode.counts.predictions} · ${fmt(row.latency_ms)} ms`;
      $("record").append(detail("Input record · observation", row.input), detail("Output record · action chunk", row.output));
    };
    const updateEpisode = () => {const episode = data.tasks[Number(taskSelect.value)].episodes[Number(episodeSelect.value)]; recordSlider.max = Math.max(0, (episode?.records.length || 0) - 1); recordSlider.value = 0; renderRecord();};
    const updateTask = () => {episodeSelect.replaceChildren(); data.tasks[Number(taskSelect.value)].episodes.forEach((episode,i) => {const option = node("option", `${episode.rollout_index} · ${episode.terminal_status} · ${episode.episode_id}`); option.value = i; episodeSelect.append(option);}); updateEpisode();};
    taskSelect.addEventListener("change", updateTask); episodeSelect.addEventListener("change", updateEpisode); recordSlider.addEventListener("input", renderRecord); updateTask();
  } else $("batch").append(node("p", "No finalized policy records are available."));
  const flatten = (value, prefix = "", rows = []) => {
    if (value != null && typeof value === "object" && !Array.isArray(value)) {
      if (!Object.keys(value).length) rows.push([prefix, "Unavailable — not recorded"]);
      Object.entries(value).forEach(([key,item]) => flatten(item, prefix ? prefix + "." + key : key, rows));
    } else rows.push([prefix, value == null ? "Unavailable — not recorded" : typeof value === "object" ? JSON.stringify(value) : fmt(value)]);
    return rows;
  };
  const parameters = flatten(data.hyperparameters);
  function renderParameters() {const query = $("parameter-search").value.toLowerCase(); $("parameter-rows").replaceChildren(); parameters.filter(row => row.join(" ").toLowerCase().includes(query)).forEach(row => {const tr = node("tr"); row.forEach(value => tr.append(node("td", value))); $("parameter-rows").append(tr);});}
  $("parameter-search").addEventListener("input", renderParameters); renderParameters();

  // Each series retains its actual canonical X coordinate. Missing values break lines.
  function chart(title, points, xLabel, unit) {
    const e = node("div", null, "card chart-card"); e.append(node("h3", title));
    if(points.length>2000){const total=points.length;points=Array.from({length:2000},(_,i)=>points[Math.round(i*(total-1)/1999)]);e.append(node("p",`Showing 2,000 ordered samples of ${fmt(total)}; original X coordinates are preserved. Full values remain in dashboard data.`,"caption"));}
    const available = points.filter(p => finite(p[0]) && finite(p[1]));
    if (!available.length) {e.append(node("p", "Unavailable — no recorded samples.", "caption")); $("charts").append(e); return;}
    const fullMin = Math.min(...available.map(p => p[0])), fullMax = Math.max(fullMin + 1, ...available.map(p => p[0]));
    let xmin = fullMin, xmax = fullMax, dragging = null;
    const controls = node("div", null, "chart-toolbar"), reset = node("button", "Reset"), canvas = node("canvas"), tip = node("div", "Hover: exact values · wheel: zoom · drag: pan", "tooltip");
    controls.append(node("span", `${xLabel} / ${unit}`), reset); e.append(controls, canvas, tip); $("charts").append(e);
    const ctx = canvas.getContext("2d"), W = 640, H = 240, left = 64, right = 20, top = 15, bottom = 40;
    canvas.width = W * 2; canvas.height = H * 2; ctx.scale(2, 2);
    const visible = () => available.filter(p => p[0] >= xmin && p[0] <= xmax);
    const transform = () => {const values = visible().map(p => p[1]); let ymin = values.length ? Math.min(...values) : 0, ymax = values.length ? Math.max(...values) : 1; const pad = ymax === ymin ? Math.max(.1, Math.abs(ymin)*.05) : (ymax-ymin)*.08; ymin -= pad; ymax += pad; return {x: x => left+(x-xmin)/(xmax-xmin)*(W-left-right), y: y => H-bottom-(y-ymin)/(ymax-ymin)*(H-top-bottom), ymin, ymax};};
    const draw = () => {
      const t = transform(); ctx.clearRect(0,0,W,H); ctx.font = "11px system-ui"; ctx.lineWidth = 1;
      for (let i=0;i<5;i++) {const y = top+i*(H-top-bottom)/4; ctx.strokeStyle="#e0e8ed";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-right,y);ctx.stroke();ctx.fillStyle="#607889";ctx.textAlign="right";ctx.fillText(fmt(t.ymax-i*(t.ymax-t.ymin)/4),left-8,y+4);}
      ctx.textAlign="center"; for(let i=0;i<5;i++) {const x=xmin+i*(xmax-xmin)/4;ctx.fillText(fmt(x),t.x(x),H-bottom+20);}
      ctx.save();ctx.beginPath();ctx.rect(left,top,W-left-right,H-top-bottom);ctx.clip();ctx.strokeStyle="#147f86";ctx.fillStyle="#147f86";ctx.lineWidth=2;ctx.beginPath();let segment=false;
      points.forEach(p => {if(!finite(p[1])){segment=false;return;} if(segment)ctx.lineTo(t.x(p[0]),t.y(p[1]));else ctx.moveTo(t.x(p[0]),t.y(p[1]));segment=true;});ctx.stroke();
      const shown=visible();if(shown.length<100)shown.forEach(p=>{ctx.beginPath();ctx.arc(t.x(p[0]),t.y(p[1]),3,0,Math.PI*2);ctx.fill();});ctx.restore();
    };
    const pos = event => (event.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*W;
    const clamp = () => {const span=Math.min(fullMax-fullMin,xmax-xmin); xmin=Math.max(fullMin,Math.min(xmin,fullMax-span));xmax=xmin+span;};
    const restore = () => {xmin=fullMin;xmax=fullMax;draw();}; reset.addEventListener("click",restore);canvas.addEventListener("dblclick",restore);
    canvas.addEventListener("wheel", event => {event.preventDefault();const ratio=Math.max(0,Math.min(1,(pos(event)-left)/(W-left-right))), anchor=xmin+ratio*(xmax-xmin), span=Math.max((fullMax-fullMin)/1000,Math.min(fullMax-fullMin,(xmax-xmin)*(event.deltaY>0?1.2:.8)));xmin=anchor-ratio*span;xmax=xmin+span;clamp();draw();},{passive:false});
    canvas.addEventListener("pointerdown",event=>{dragging={x:pos(event),min:xmin,max:xmax};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener("pointerup",()=>{dragging=null;});canvas.addEventListener("pointercancel",()=>{dragging=null;});
    canvas.addEventListener("pointermove",event=>{if(dragging){const shift=(pos(event)-dragging.x)/(W-left-right)*(dragging.max-dragging.min);xmin=dragging.min-shift;xmax=dragging.max-shift;clamp();draw();}const x=xmin+(pos(event)-left)/(W-left-right)*(xmax-xmin), shown=visible();if(shown.length){const nearest=shown.reduce((a,b)=>Math.abs(a[0]-x)<Math.abs(b[0]-x)?a:b);tip.textContent=`${xLabel}: ${fmt(nearest[0])} · ${fmt(nearest[1])} ${unit}`;}});draw();
  }
  if (train) {
    const rows = data.metrics, series = key => rows.map((row,i) => [row.global_step ?? row.optimizer_step ?? i+1, row[key] ?? null]);
    chart("Training loss · recorded microbatch",series("training_loss"),"Optimizer step","loss");
    if (chunkTrain) chart("Valid action steps",series("valid_action_steps"),"Optimizer step","steps");
    else chart("Supervised action-token accuracy",series("action_token_accuracy"),"Optimizer step","fraction");
    chart("Optimizer step duration",series("step_duration_ms"),"Optimizer step","ms");
    chart("GPU allocator memory",series("gpu_memory_allocated_bytes").map(([x,y])=>[x,y==null?null:y/1048576]),"Optimizer step","MiB");
    $("outcome").append(node("p", "Robot task success: unavailable in a training-only run. Evaluation is reported in the corresponding eval dashboard.", "notes"));
    $("metric-table").append(detail("Model parameter counts",data.parameter_counts),detail("Performance statistics",data.performance.statistics));
  } else {
    const outcome=data.outcome;
    if(outcome){const total=outcome.eligible_episode_count, success=outcome.successful_episode_count;$("outcome").append(node("p",total===1?`Binary episode outcome: ${success===1?"success":"non-success"}. One episode does not establish a statistical success rate.`:`Success rate: ${total?fmt(success/total*100)+"%":"Unavailable"} · ${success} / ${total} finalized episodes. Non-success: ${total-success} / ${total}. Missing: ${outcome.missing_episode_count}.`)); const bar=node("div",null,"progress"),fill=node("span");fill.style.width=`${total?success/total*100:0}%`;bar.append(fill);$("outcome").append(bar);}
    const episodes=data.tasks.flatMap(t=>t.episodes);
    chart("Episode outcome",episodes.map((ep,i)=>[i+1,ep.success?1:0]),"Episode index","0 / 1");
    chart("Executed actions per episode",episodes.map((ep,i)=>[i+1,ep.counts.executed_actions]),"Episode index","actions");
    chart("Mean policy inference latency",episodes.map((ep,i)=>[i+1,ep.trace_view.inference_latency_statistics_ms.mean]),"Episode index","ms");
    if(data.scope_label) {
      const options=node("select");["tx","ty","tz","rx","ry","rz","gripper"].forEach((name,i)=>{const option=node("option",name);option.value=i;options.append(option);});
      const host=node("div");host.append(node("p","Applied-action component · samples are ordered within each episode", "caption"),options);$("charts").append(host);
      let previous=null;const drawActions=()=>{if(previous)previous.remove();const rows=episodes.flatMap(ep=>ep.records.flatMap(r=>r.output.executed.map(a=>a.applied)));chart(`Applied action · ${options.selectedOptions[0].textContent}`,rows.map((row,i)=>[i,row[Number(options.value)]??null]),"Executed sample index","command (see action_spec units)");previous=$("charts").lastChild;};options.addEventListener("change",drawActions);drawActions();
    }
    const table=node("table"),head=node("tr");["Task","Rollout","Seed","Status","Calls","Applied actions","Reason"].forEach(x=>head.append(node("th",x)));table.append(head);
    data.tasks.forEach(task=>task.episodes.forEach(ep=>{const tr=node("tr");[task.task_id,ep.rollout_index,ep.seed,ep.terminal_status,ep.counts.predictions,ep.counts.executed_actions,ep.terminal_reason].forEach(x=>tr.append(node("td",fmt(x))));table.append(tr);}));$("metric-table").append(table);
    const metrics=node("table"),metricHead=node("tr");["Scope","Metric","Value","Unit","n","Min","Median","Mean","Sample SD","Max","Status / reason"].forEach(x=>metricHead.append(node("th",x)));metrics.append(metricHead);
    const metricRows=(scope,rows,episodeCount)=>rows.forEach(metric=>{if(metric.metric_id==="task.success_rate" && episodeCount===1)return;const tr=node("tr"),stats=metric.statistics||{},value=typeof metric.value==="object"&&metric.value!=null?"Aggregate":fmt(metric.value);[scope,metric.metric_id,value,metric.unit,stats.valid_n,stats.minimum,stats.median,stats.mean,stats.standard_deviation,stats.maximum,`${metric.status}${metric.reason?" · "+metric.reason:""}`].forEach(value=>tr.append(node("td",fmt(value))));metrics.append(tr);});
    data.tasks.forEach(task=>{metricRows(task.task_id,task.metrics,task.episodes.length);task.episodes.forEach(ep=>metricRows(ep.episode_id,ep.metrics,1));});$("metric-table").append(node("h3","Canonical task and episode metrics"),metrics);
  }
})();
