"use strict";
(() => {
  const data = JSON.parse(document.getElementById("dashboard-data").textContent);
  const $ = id => document.getElementById(id);
  const node = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  const fmt = value => value == null ? "Unavailable" : typeof value === "number" ? value.toLocaleString(undefined, {maximumSignificantDigits: 7}) : String(value);
  const exactValue = value => value == null ? "Unavailable" : typeof value === "object" ? JSON.stringify(value) : String(value);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const card = (title, value, caption) => { const e = node("div", null, "card"); e.append(node("div", title, "caption"), node("div", fmt(value), "kpi")); if (caption) e.append(node("p", caption, "caption"));if(value==null||value==="Unavailable"||(["Missing episodes","Non-successful episodes"].includes(title)&&value>0))e.classList.add("needs-attention");return e; };
  const train = data.mode === "train";
  const settings = data.hyperparameters.training || {};
  const evidence = data.evidence;
  const chunkTrain = train && evidence?.objective === "masked_l1";
  const countLabels = {tasks:"Tasks", episodes:"Finalized episodes", observations:"Observations", predictions:"Policy calls", predicted_actions:"Predicted actions", executed_actions:"Applied actions", optimizer_steps:"Recorded optimizer steps", examples_seen:"Examples seen", dataset_records:"Dataset records", supervised_action_tokens:"Supervised action tokens", microbatches:"Microbatches"};
  Object.entries(data.counts).forEach(([key, value]) => $("kpis").append(card(countLabels[key] || key.replaceAll("_", " "), value)));
  data.notes.forEach(text => $("notes").append(node("p", text)));
  const lastValue = key => data.latest_metrics?.[key] ?? [...(data.metrics || [])].reverse().find(row => finite(row[key]))?.[key];
  if (train) {
    $("highlights").append(card("Latest recorded loss", lastValue("training_loss"), "Last recorded microbatch; lower values alone do not establish robot task success."));
    $("highlights").append(chunkTrain ? card("Latest valid action steps", lastValue("valid_action_steps")) : card("Latest action-token accuracy", finite(lastValue("action_token_accuracy")) ? fmt(lastValue("action_token_accuracy")*100)+"%" : null, "Supervised tokens, separate from robot task success."));
    $("highlights").append(card("Batch evidence", evidence ? "Captured" : "Unavailable", evidence ? "Actual forward-pass tensors and first-record previews below." : "No forward-pass capture was recorded for this run."));
  } else {
    const outcome = data.outcome;
    if (outcome) {
      $("highlights").append(card("Successful episodes", outcome.successful_episode_count), card("Non-successful episodes", outcome.eligible_episode_count-outcome.successful_episode_count), card("Missing episodes", outcome.missing_episode_count, "Missing episodes are separate from finalized non-success outcomes."));
    }
  }
  // A contract schematic, with captured evidence rather than invented architecture.
  const stages = [
    ["inputs", "Inputs", train ? `${fmt(data.counts.dataset_records)} dataset records; ${fmt(data.counts.examples_seen)} examples seen, including repeats. Image tensors and prompt tokens enter each microbatch.` : `${fmt(data.counts.observations)} observations with camera images, instruction and proprioception; ${fmt(data.counts.predictions)} policy calls.`],
    ["representation", train ? "Batch & supervision" : "Observation contract", train ? `Microbatch size: ${fmt(settings.per_device_batch_size)}. ${chunkTrain ? "Continuous future actions and validity mask" : "Attention mask and supervised action-token labels"}. Exact tensor shapes are shown in the explorer.` : "One observation per canonical call. Camera shapes, image references, instruction and state values are recorded in each input record."],
    ["policy", "Model / policy", train ? `Precision: ${fmt(settings.precision)}. Gradient accumulation: ${fmt(settings.gradient_accumulation_steps)} microbatches per optimizer step. Model settings and parameter counts remain visible below.` : "The configured policy maps the observation to an action chunk. Internal layers, tokenization and tensor batching are unavailable unless captured in policy metadata."],
    ["outputs", train ? "Prediction & objective" : "Predicted action chunk", train ? (chunkTrain ? "Continuous action chunks and masked L1 loss. Captured predictions, targets and valid steps are shown below." : "Next-token logits and supervised loss. Captured predictions, targets and token accuracy are shown below.") : `${fmt(data.counts.predictions)} policy calls produce ${fmt(data.counts.predicted_actions)} predicted actions. Horizon, dtype, action specification and units remain visible in each record.`],
    ["result", train ? "Optimizer & checkpoint" : "Applied action & outcome", train ? `${fmt(data.counts.optimizer_steps)} recorded optimizer steps. Training metrics describe optimization; robot task success requires a separate evaluation.` : `${fmt(data.counts.executed_actions)} actions applied to the environment. Requested and applied commands remain separate; finalized episode outcomes appear above.`]
  ];
  const svgNS = "http://www.w3.org/2000/svg";
  const svgNode = (tag, attrs, text) => {const el=document.createElementNS(svgNS,tag);Object.entries(attrs||{}).forEach(([key,value])=>el.setAttribute(key,value));if(text)el.textContent=text;return el;};
  const svg=svgNode("svg", {viewBox:"0 0 1000 190",role:"img","aria-label":"Model data flow: inputs, representation, policy, predictions and measured results"});
  svg.append(svgNode("title",{},"Recorded model data flow"));
  const groups=[],connections=[];
  stages.forEach(([id,title],i)=>{
    const x=12+i*199;
    if(i<4) {const connection=svgNode("path",{d:`M ${x+178} 96 H ${x+194} m -5 -5 l 5 5 -5 5`,class:"model-link"});svg.append(connection);connections.push(connection);}
    const group=svgNode("g",{class:"model-stage","data-stage":id});
    group.append(svgNode("rect",{x,y:24,width:178,height:139,rx:12,class:"stage-frame"}));
    if(i===2){
      for(let layer=0;layer<3;layer++)for(let row=0;row<3;row++){
        if(layer<2)for(let next=0;next<3;next++)group.append(svgNode("line",{x1:x+53+layer*36,y1:51+row*24,x2:x+89+layer*36,y2:51+next*24,class:"neural-link"}));
      }
      for(let layer=0;layer<3;layer++)for(let row=0;row<3;row++)group.append(svgNode("circle",{cx:x+53+layer*36,cy:51+row*24,r:6,class:"neural-node"}));
    } else if(i===0){
      group.append(svgNode("rect",{x:x+46,y:48,width:86,height:52,rx:5,class:"signal-shape"}),svgNode("path",{d:`M ${x+53} 91 l 22 -24 16 16 15 -12 18 20`,class:"signal-line"}),svgNode("circle",{cx:x+111,cy:61,r:5,class:"neural-node"}));
    } else {
      for(let j=0;j<4;j++)group.append(svgNode("rect",{x:x+49+j*21,y:102-[29,53,42,64][j],width:12,height:[29,53,42,64][j],rx:3,class:"signal-shape"}));
    }
    const label=svgNode("text",{x:x+89,y:139,"text-anchor":"middle",class:"stage-label"});
    if(title.length>18){const words=title.split(" "),middle=Math.ceil(words.length/2);label.append(svgNode("tspan",{x:x+89,y:128},words.slice(0,middle).join(" ")),svgNode("tspan",{x:x+89,y:148},words.slice(middle).join(" ")));}else label.textContent=title;
    group.append(label);svg.append(group);groups.push(group);
  });
  $("model-illustration").append(svg);
  const stageButtons=[];
  stages.forEach(([id,title,body],i)=>{
    const article=node("article",null,"flow-stage"),button=node("button",title,"stage-button");button.type="button";button.setAttribute("aria-pressed","false");
    button.addEventListener("click",()=>selectStage(i,true));article.append(button,node("p",body));
    const group=(data.key_parameters||[]).find(group=>group.stage===id);
    if(group){const list=node("dl",null,"stage-parameters");group.items.forEach(item=>{const entry=node("div",null,"stage-parameter");entry.append(node("dt",item.label),node("dd",exactValue(item.value),item.value==null?"stage-parameter-value unavailable":"stage-parameter-value"),node("dd",item.source||"Not recorded","stage-parameter-source"));list.append(entry);});article.append(list);}
    $("flow-content").append(article);stageButtons.push(button);
    groups[i].addEventListener("click",()=>selectStage(i,true));
  });
  function selectStage(index,scroll=false){connections.forEach((connection,i)=>connection.classList.toggle("selected",i===index||i===index-1));if(scroll){const host=$("model-illustration"),x=(12+index*199)/1000*svg.getBoundingClientRect().width;host.scrollLeft=Math.max(0,x-(host.clientWidth-178/1000*svg.getBoundingClientRect().width)/2);}groups.forEach((group,i)=>group.classList.toggle("selected",i===index));stageButtons.forEach((button,i)=>{button.setAttribute("aria-pressed",String(i===index));button.parentElement.classList.toggle("selected",i===index);});}
  selectStage(2);
  const detail = (title, value) => {
    const e = node("div", null, "card"); e.append(node("h3", title));
    if (value == null) {e.append(node("p", "Unavailable — this evidence was not recorded.")); return e;}
    const entries = typeof value === "object" && !Array.isArray(value) ? Object.entries(value) : [["Values", value]];
    const priority=["instruction","images","processed_image_preview","actions","executed","loss","predicted_tokens","targets","input_ids","labels","shape","dtype"];
    entries.sort(([a],[b])=>(priority.includes(a)?priority.indexOf(a):priority.length)-(priority.includes(b)?priority.indexOf(b):priority.length));
    entries.forEach(([key, item]) => {
      if (key === "images") {
        const images = node("div", null, "image-row");
        item.forEach(image => {const figure = node("figure"); if (image.preview) {const img = node("img"); img.src = image.preview; img.alt = image.name + " canonical RGB preview"; figure.append(img);} figure.append(node("figcaption", `${image.name} · ${image.shape.join(" × ")} · ${image.dtype}`)); if (!image.preview) figure.append(node("p", "RGB preview not embedded for this record; array reference is retained.", "caption")); images.append(figure);});
        e.append(images);
      }
      if (key === "processed_image_preview" && item?.values?.length) {
        const pixels=item.values, h=pixels.length, w=pixels[0].length, canvas=node("canvas"), values=pixels.flat(2).filter(finite);
        const lo=Math.min(...values), hi=Math.max(...values), span=hi-lo||1;
        canvas.setAttribute("role","img");canvas.setAttribute("aria-label",item.semantics||"Rescaled processed image tensor preview");canvas.width=w;canvas.height=h;canvas.style.width="192px";canvas.style.imageRendering="pixelated";
        const ctx=canvas.getContext("2d"), image=ctx.createImageData(w,h);
        for(let y=0;y<h;y++)for(let x=0;x<w;x++){for(let c=0;c<3;c++)image.data[(y*w+x)*4+c]=Math.round(((pixels[y][x][c]??lo)-lo)/span*255);image.data[(y*w+x)*4+3]=255;}ctx.putImageData(image,0,0);
        e.append(canvas,node("p",item.semantics,"caption"));
      }
      const d = node("details"); const summary = node("summary", key.replaceAll("_", " ")); d.append(summary);
      if (typeof item === "object" && item != null) {
        if(key==="statistics"){
          const table=node("table"),head=node("tr");["Series / statistic","Recorded value"].forEach(value=>head.append(node("th",value)));table.append(head);
          const addRows=(value,path)=>{if(value!=null&&typeof value==="object"&&!Array.isArray(value))Object.entries(value).forEach(([name,value])=>addRows(value,path?path+" / "+name:name));else{const row=node("tr");row.append(node("td",path),node("td",exactValue(value)));table.append(row);}};addRows(item,"");
          const wrap=node("div",null,"table-scroll");wrap.tabIndex=0;wrap.setAttribute("role","region");wrap.setAttribute("aria-label",title+" statistics");wrap.append(table);d.append(wrap);
        } else {const pre=node("pre",JSON.stringify(item,(k,v)=>k==="preview"?"[embedded image]":v,2));pre.tabIndex=0;pre.setAttribute("aria-label",key.replaceAll("_"," ")+" recorded data");d.append(pre);}
      } else d.append(node("p",exactValue(item)));
      d.open = true;
      e.append(d);
    });
    return e;
  };
  // Script chunks work offline when reports are opened directly with file://.
  const pendingChunks = new Map();
  const deliver = (key, payload) => {const request=pendingChunks.get(key);if(request){pendingChunks.delete(key);request.script.remove();request.resolve(payload);}};
  window.ovlabReceiveEpisode = (key, episode) => deliver("episode:"+key, episode);
  window.ovlabReceiveTraining = payload => deliver("training", payload);
  function loadChunk(path, kind) {
    const key=kind==="training" ? kind : "episode:"+path.split("/").pop().replace(/\.js$/, "");
    if(pendingChunks.has(key)) return pendingChunks.get(key).promise;
    const script=node("script");script.src=path;
    let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
    pendingChunks.set(key,{script,promise,resolve,reject});
    const fail=message=>{const request=pendingChunks.get(key);if(request){pendingChunks.delete(key);script.remove();request.reject(new Error(message));}};
    script.onerror=()=>fail("Could not load the selected data file. Keep the complete report folder together.");
    script.onload=()=>fail("The selected data file did not contain the expected report data.");
    document.head.append(script);return promise;
  }
  let inspectEpisode=()=>{}, renderSelectedActions=()=>{};
  if (train) {
    if(data.training_script){
      const load=node("button","Load training results"), status=node("p","Training metrics and captured batch details load when requested.");
      status.setAttribute("role","status");$("batch").append(status,load);
      load.addEventListener("click",async()=>{load.disabled=true;status.textContent="Loading training results…";try{const payload=await loadChunk(data.training_script,"training");data.metrics=payload.metrics;data.evidence=payload.evidence;$("batch").replaceChildren();renderTrainingEvidence(payload.evidence);renderTrainingPerformance();}catch(error){status.textContent=error.message;load.disabled=false;}});
    } else renderTrainingEvidence(evidence);
  } else if (data.tasks.length) {
    const taskSelect = node("select"), episodeSelect = node("select"), recordSlider = node("input");
    const load=node("button","Load run details"), previous=node("button","Previous call"),next=node("button","Next call");
    const contextHost=node("div",null,"episode-context");$("explorer").append(contextHost);
    let selectedEpisode=null,requestVersion=0;
    recordSlider.type="range";recordSlider.min="0";recordSlider.step="1";
    const recordLabel=node("label","Prediction / policy call");recordLabel.append(recordSlider);
    for(const [label,control] of [["Task",taskSelect],["Episode / run",episodeSelect]]){const l=node("label",label);l.append(control);$("selectors").append(l);}
    $("selectors").append(load,recordLabel,previous,next);
    data.tasks.forEach((task,i)=>{const option=node("option",task.task_id);option.value=i;taskSelect.append(option);});
    const summary=()=>data.tasks[Number(taskSelect.value)]?.episodes[Number(episodeSelect.value)];
    function renderRecord(){
      $("record").replaceChildren();
      const episode=selectedEpisode,row=episode?.records[Number(recordSlider.value)];
      previous.disabled=!episode?.records.length||Number(recordSlider.value)===0;next.disabled=!episode?.records.length||Number(recordSlider.value)>=episode.records.length-1;recordSlider.disabled=!episode?.records.length;
      if(!episode){recordLabel.firstChild.textContent="Prediction / policy call";return;}
      if(!row){recordLabel.firstChild.textContent="No recorded policy calls";$("record").append(detail("Policy input",null),detail("Policy output",null));return;}
      recordLabel.firstChild.textContent=`Prediction ${row.index+1} / ${episode.counts.predictions} · ${fmt(row.latency_ms)} ms`;
      recordSlider.setAttribute("aria-valuetext",`Prediction ${row.index+1} of ${episode.counts.predictions}`);
      $("record").append(detail("Input record · observation",row.input),detail("Output record · action chunk",row.output));
    }
    function updateEpisode(){
      requestVersion++;selectedEpisode=null;recordSlider.value=0;recordSlider.max=0;contextHost.replaceChildren();renderRecord();renderSelectedActions(null);
      const episode=summary();load.disabled=!episode;$("batch").replaceChildren(node("p",episode ? `Rollout ${episode.rollout_index} · seed ${episode.seed} · ${episode.terminal_status} · ${episode.counts.predictions} calls · ${episode.counts.executed_actions} applied actions. Select Load run details to inspect this episode.` : "No finalized episodes in this task."));
    }
    function updateTask(){episodeSelect.replaceChildren();data.tasks[Number(taskSelect.value)].episodes.forEach((episode,i)=>{const option=node("option",`${episode.rollout_index} · ${episode.terminal_status} · ${episode.episode_id}`);option.value=i;episodeSelect.append(option);});updateEpisode();}
    async function loadSelected(){
      const episode=summary();if(!episode)return;
      const version=++requestVersion;load.disabled=true;$("batch").replaceChildren(node("p","Loading selected run details…"));
      try{
        const loaded=episode.detail_script?await loadChunk(episode.detail_script,"episode"):episode;
        if(version!==requestVersion)return;
        selectedEpisode=loaded;recordSlider.max=Math.max(0,loaded.records.length-1);recordSlider.value=0;load.disabled=false;
        $("batch").replaceChildren(node("p",`Rollout ${loaded.rollout_index} · seed ${loaded.seed} · ${loaded.terminal_status} · ${loaded.counts.predictions} calls · ${loaded.counts.executed_actions} applied actions. Every recorded policy call is selectable.`));
        contextHost.replaceChildren();const {records,...context}=loaded;contextHost.append(detail(`Episode context and metrics · ${loaded.episode_id}`,context));
        renderRecord();renderSelectedActions(loaded);
      }catch(error){if(version!==requestVersion)return;load.disabled=false;$("batch").replaceChildren(node("p",error.message));}
    }
    inspectEpisode=(taskIndex,episodeIndex)=>{taskSelect.value=taskIndex;updateTask();episodeSelect.value=episodeIndex;updateEpisode();$("explorer").scrollIntoView();loadSelected();};
    const step=delta=>{recordSlider.value=Number(recordSlider.value)+delta;renderRecord();};
    load.addEventListener("click",loadSelected);previous.addEventListener("click",()=>step(-1));next.addEventListener("click",()=>step(1));taskSelect.addEventListener("change",updateTask);episodeSelect.addEventListener("change",updateEpisode);recordSlider.addEventListener("input",renderRecord);updateTask();
  } else $("batch").append(node("p", "No finalized policy records are available."));
  function renderTrainingEvidence(captured){
    $("batch").append(node("p",captured?`Actual captured microbatch ${captured.microbatch_index}; ${fmt(captured.batch_size)} records. Tensor shapes describe the full batch. First-record previews are bounded; total element counts and truncation are explicit.`:"No batch capture exists in this historical run. Hyperparameters show the configured batch size; actual examples are unavailable."));
    $("record").append(detail("Input batch",captured?.input_batch),detail("Input record · index 0",captured?.input_record),detail("Output batch",captured?.output_batch),detail("Output record · index 0",captured?.output_record));
  }
  const flatten = (value, prefix = "", rows = []) => {
    if (value != null && typeof value === "object" && !Array.isArray(value)) {
      if (!Object.keys(value).length) rows.push([prefix, "Unavailable — not recorded"]);
      Object.entries(value).forEach(([key,item]) => flatten(item, prefix ? prefix + "." + key : key, rows));
    } else rows.push([prefix, value == null ? "Unavailable — not recorded" : exactValue(value)]);
    return rows;
  };
  const parameters = flatten(data.hyperparameters);
  function renderParameters() {
    const query = $("parameter-search").value.toLowerCase().trim(), rows=parameters.filter(row => row.join(" ").toLowerCase().includes(query));
    $("parameter-rows").replaceChildren();
    rows.forEach(row => {const tr = node("tr"); row.forEach(value => tr.append(node("td", value))); $("parameter-rows").append(tr);});
    $("parameter-count").textContent=`${rows.length} of ${parameters.length} settings shown`;
    if(!rows.length){const row=node("tr"),cell=node("td","No matching settings. Clear the search to see all recorded hyperparameters.");cell.colSpan=2;row.append(cell);$("parameter-rows").append(row);}
  }
  $("parameter-search").addEventListener("input", renderParameters); renderParameters();

  // Each series retains its actual canonical X coordinate. Missing values break lines.
  function chart(title, points, xLabel, unit) {
    const e = node("div", null, "card chart-card"); e.append(node("h3", title));
    const recordedPoints=points;
    const recordedValues=recordedPoints.filter(p=>finite(p[0])&&finite(p[1]));
    if(recordedValues.length){
      const values=recordedValues.map(p=>p[1]);
      let minimum=Infinity,maximum=-Infinity;values.forEach(value=>{minimum=Math.min(minimum,value);maximum=Math.max(maximum,value);});
      e.append(node("p",`Latest: ${fmt(values[values.length-1])} ${unit} · Minimum: ${fmt(minimum)} · Maximum: ${fmt(maximum)} · ${fmt(values.length)} valid / ${fmt(recordedPoints.length)} recorded samples`,"chart-summary"));
    }
    const available=recordedValues;
    const canonicalIndex=new Map(recordedPoints.map((point,index)=>[point,index]));
    const missingBefore=new Uint32Array(recordedPoints.length+1);
    recordedPoints.forEach((point,index)=>{missingBefore[index+1]=missingBefore[index]+(!finite(point[0])||!finite(point[1])?1:0);});
    points=available;
    if(available.length>2000){points=Array.from({length:2000},(_,i)=>available[Math.round(i*(available.length-1)/1999)]);e.append(node("p",`Plot shows 2,000 of ${fmt(available.length)} finite samples; original X coordinates and missing-data gaps are preserved. Every recorded sample remains in the table below.`,"caption"));}
    if (!available.length) e.append(node("p", "Unavailable — no finite recorded samples. Missing values remain listed in the recorded series.", "caption"));
    if (!recordedPoints.length) {$("charts").append(e);return;}
    let fullMin=Infinity,fullMax=-Infinity;available.forEach(point=>{fullMin=Math.min(fullMin,point[0]);fullMax=Math.max(fullMax,point[0]);});if(!available.length){fullMin=0;fullMax=1;}fullMax=Math.max(fullMin+1,fullMax);
    let xmin = fullMin, xmax = fullMax, dragging = null;
    const controls = node("div", null, "chart-toolbar"), reset = node("button", "Reset view"), zoomIn=node("button","Zoom in"),zoomOut=node("button","Zoom out"), canvas = node("canvas"), tip = node("output", "Hover for exact values. Wheel to zoom; drag to pan.", "tooltip");
    const actions=node("div",null,"chart-actions");actions.append(zoomIn,zoomOut,reset);controls.append(node("span", `${xLabel} / ${unit}`), actions);
    canvas.setAttribute("role","img");canvas.setAttribute("aria-label",`${title}. Exact values and full recorded series follow the chart.`);
    const sampleLabel=node("label","Inspect an exact recorded sample"),sample=node("input");sample.type="range";sample.min="0";sample.max=recordedPoints.length-1;sample.value="0";sample.step="1";sampleLabel.append(sample);
    const exact=node("output",null,"exact-sample");exact.setAttribute("aria-live","polite");
    const inspect=()=>{const point=recordedPoints[Number(sample.value)];exact.textContent=`Sample ${Number(sample.value)+1} / ${recordedPoints.length} — ${xLabel}: ${exactValue(point[0])} · ${exactValue(point[1])} ${unit}`;sample.setAttribute("aria-valuetext",exact.textContent);};sample.addEventListener("input",inspect);inspect();
    e.append(controls,canvas,tip,sampleLabel,exact);
    const series=node("details",null,"series-data");series.open=true;series.append(node("summary",`Recorded series · all ${fmt(recordedPoints.length)} samples`));
    const wrap=node("div",null,"table-scroll series-scroll"),table=node("table"),thead=node("thead"),head=node("tr"),tbody=node("tbody");wrap.tabIndex=0;wrap.setAttribute("role","region");wrap.setAttribute("aria-label",`${title} recorded series`);
    [xLabel,unit].forEach(text=>{const th=node("th",text);th.scope="col";head.append(th);});thead.append(head);table.append(thead,tbody);
    const pageSize=100,previousPage=node("button","Previous samples"),nextPage=node("button","Next samples"),pageStatus=node("span",null,"caption"),pager=node("div",null,"pagination");let page=0;
    function renderPage(){tbody.replaceChildren();const start=page*pageSize,end=Math.min(start+pageSize,recordedPoints.length),fragment=document.createDocumentFragment();for(let i=start;i<end;i++){const row=node("tr");recordedPoints[i].forEach(value=>row.append(node("td",exactValue(value))));fragment.append(row);}tbody.append(fragment);pageStatus.textContent=`Samples ${start+1}–${end} of ${recordedPoints.length}`;previousPage.disabled=page===0;nextPage.disabled=end>=recordedPoints.length;}
    previousPage.addEventListener("click",()=>{page--;renderPage();});nextPage.addEventListener("click",()=>{page++;renderPage();});renderPage();pager.append(previousPage,pageStatus,nextPage);wrap.append(table);series.append(wrap,pager);e.append(series);$("charts").append(e);
    const ctx = canvas.getContext("2d"), W = 640, H = 240, left = 64, right = 20, top = 15, bottom = 40;
    canvas.width = W * 2; canvas.height = H * 2; ctx.scale(2, 2);
    const visible = () => available.filter(p => p[0] >= xmin && p[0] <= xmax);
    const transform = () => {const values=visible();let ymin=Infinity,ymax=-Infinity;values.forEach(point=>{ymin=Math.min(ymin,point[1]);ymax=Math.max(ymax,point[1]);});if(!values.length){ymin=0;ymax=1;} const pad = ymax === ymin ? Math.max(.1, Math.abs(ymin)*.05) : (ymax-ymin)*.08; ymin -= pad; ymax += pad; return {x: x => left+(x-xmin)/(xmax-xmin)*(W-left-right), y: y => H-bottom-(y-ymin)/(ymax-ymin)*(H-top-bottom), ymin, ymax};};
    const draw = () => {
      const t = transform(); ctx.clearRect(0,0,W,H); ctx.font = "11px system-ui"; ctx.lineWidth = 1;
      for (let i=0;i<5;i++) {const y = top+i*(H-top-bottom)/4; ctx.strokeStyle="#e0e8ed";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-right,y);ctx.stroke();ctx.fillStyle="#607889";ctx.textAlign="right";ctx.fillText(fmt(t.ymax-i*(t.ymax-t.ymin)/4),left-8,y+4);}
      ctx.textAlign="center"; for(let i=0;i<5;i++) {const x=xmin+i*(xmax-xmin)/4;ctx.fillText(fmt(x),t.x(x),H-bottom+20);}
      ctx.save();ctx.beginPath();ctx.rect(left,top,W-left-right,H-top-bottom);ctx.clip();ctx.strokeStyle="#147f86";ctx.fillStyle="#147f86";ctx.lineWidth=2;ctx.beginPath();let previousIndex=null;
      points.forEach(p=>{const index=canonicalIndex.get(p),connected=previousIndex!==null&&missingBefore[index]===missingBefore[previousIndex+1];if(connected)ctx.lineTo(t.x(p[0]),t.y(p[1]));else ctx.moveTo(t.x(p[0]),t.y(p[1]));previousIndex=index;});ctx.stroke();
      const shown=visible();if(shown.length<100)shown.forEach(p=>{ctx.beginPath();ctx.arc(t.x(p[0]),t.y(p[1]),3,0,Math.PI*2);ctx.fill();});ctx.restore();
    };
    const pos = event => (event.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*W;
    const clamp = () => {const span=Math.min(fullMax-fullMin,xmax-xmin); xmin=Math.max(fullMin,Math.min(xmin,fullMax-span));xmax=xmin+span;};
    const restore = () => {xmin=fullMin;xmax=fullMax;draw();};
    const zoom=factor=>{const midpoint=(xmin+xmax)/2,span=Math.max((fullMax-fullMin)/1000,Math.min(fullMax-fullMin,(xmax-xmin)*factor));xmin=midpoint-span/2;xmax=midpoint+span/2;clamp();draw();};zoomIn.addEventListener("click",()=>zoom(.5));zoomOut.addEventListener("click",()=>zoom(2));
    reset.addEventListener("click",restore);canvas.addEventListener("dblclick",restore);
    canvas.addEventListener("wheel", event => {event.preventDefault();const ratio=Math.max(0,Math.min(1,(pos(event)-left)/(W-left-right))), anchor=xmin+ratio*(xmax-xmin), span=Math.max((fullMax-fullMin)/1000,Math.min(fullMax-fullMin,(xmax-xmin)*(event.deltaY>0?1.2:.8)));xmin=anchor-ratio*span;xmax=xmin+span;clamp();draw();},{passive:false});
    canvas.addEventListener("pointerdown",event=>{dragging={x:pos(event),min:xmin,max:xmax};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener("pointerup",()=>{dragging=null;});canvas.addEventListener("pointercancel",()=>{dragging=null;});
    canvas.addEventListener("pointermove",event=>{if(dragging){const shift=(pos(event)-dragging.x)/(W-left-right)*(dragging.max-dragging.min);xmin=dragging.min-shift;xmax=dragging.max-shift;clamp();draw();}const x=xmin+(pos(event)-left)/(W-left-right)*(xmax-xmin), shown=visible();if(shown.length){const nearest=shown.reduce((a,b)=>Math.abs(a[0]-x)<Math.abs(b[0]-x)?a:b);tip.textContent=`${xLabel}: ${exactValue(nearest[0])} · ${exactValue(nearest[1])} ${unit}`;}});draw();
  }
  function renderTrainingPerformance() {
    $("charts").replaceChildren();$("metric-table").replaceChildren();
    const rows = data.metrics, series = key => rows.map((row,i) => [row.global_step ?? row.optimizer_step ?? i+1, row[key] ?? null]);
    chart("Training loss · recorded microbatch",series("training_loss"),"Optimizer step","loss");
    if (chunkTrain) chart("Valid action steps",series("valid_action_steps"),"Optimizer step","steps");
    else chart("Supervised action-token accuracy",series("action_token_accuracy"),"Optimizer step","fraction");
    chart("Optimizer step duration",series("step_duration_ms"),"Optimizer step","ms");
    chart("GPU allocator memory",series("gpu_memory_allocated_bytes").map(([x,y])=>[x,y==null?null:y/1048576]),"Optimizer step","MiB");
    $("metric-table").append(detail("Model parameter counts",data.parameter_counts),detail("System performance evidence",data.performance));
    const table=node("table"),thead=node("thead"),head=node("tr"),tbody=node("tbody"),keys=[...new Set(data.metrics.flatMap(row=>Object.keys(row)))];
    keys.forEach(key=>{const th=node("th",key.replaceAll("_"," "));th.scope="col";head.append(th);});thead.append(head);table.append(thead,tbody);
    const pageSize=100,previousPage=node("button","Previous metrics"),nextPage=node("button","Next metrics"),pageStatus=node("span",null,"caption"),pager=node("div",null,"pagination");let page=0;
    function renderPage(){tbody.replaceChildren();const start=page*pageSize,end=Math.min(start+pageSize,data.metrics.length),fragment=document.createDocumentFragment();for(let i=start;i<end;i++){const tr=node("tr"),row=data.metrics[i];keys.forEach(key=>tr.append(node("td",exactValue(row[key]))));fragment.append(tr);}tbody.append(fragment);pageStatus.textContent=`Rows ${data.metrics.length?start+1:0}–${end} of ${data.metrics.length}`;previousPage.disabled=page===0;nextPage.disabled=end>=data.metrics.length;}
    previousPage.addEventListener("click",()=>{page--;renderPage();});nextPage.addEventListener("click",()=>{page++;renderPage();});renderPage();pager.append(previousPage,pageStatus,nextPage);
    const wrap=node("div",null,"table-scroll");wrap.tabIndex=0;wrap.setAttribute("role","region");wrap.setAttribute("aria-label","All recorded training metrics");wrap.append(table);$("metric-table").append(node("h3","All recorded training metrics"),wrap,pager);
  }
  if(train){
    $("outcome").append(node("p", "Robot task success: unavailable in a training-only run. Evaluation is reported in the corresponding eval dashboard.", "notes"));
    if(data.training_script) $("charts").append(node("p","Load training results in the explorer to view charts and metrics.","caption"));
    else renderTrainingPerformance();
  } else {
    const outcome=data.outcome;
    if(outcome){const total=outcome.eligible_episode_count, success=outcome.successful_episode_count;$("outcome").append(node("p",total===1?`Binary episode outcome: ${success===1?"success":"non-success"}. One episode does not establish a statistical success rate.`:`Success rate: ${total?fmt(success/total*100)+"%":"Unavailable"} · ${success} / ${total} finalized episodes. Non-success: ${total-success} / ${total}. Missing: ${outcome.missing_episode_count}.`)); const bar=node("div",null,"progress"),fill=node("span");bar.setAttribute("role","img");bar.setAttribute("aria-label",`${success} successful and ${total-success} non-successful finalized episodes`);fill.style.width=`${total?success/total*100:0}%`;bar.append(fill);$("outcome").append(bar);}
    const episodes=data.tasks.flatMap(t=>t.episodes);
    chart("Episode outcome",episodes.map((ep,i)=>[i+1,ep.success?1:0]),"Episode index","0 / 1");
    chart("Executed actions per episode",episodes.map((ep,i)=>[i+1,ep.counts.executed_actions]),"Episode index","actions");
    chart("Mean policy inference latency",episodes.map((ep,i)=>[i+1,ep.trace_view.inference_latency_statistics_ms.mean]),"Episode index","ms");
    const actionHost=node("div",null,"action-explorer"),actionStatus=node("p","Load a run to inspect its applied actions.","caption"),options=node("select");options.setAttribute("aria-label","Applied action component");
    ["tx","ty","tz","rx","ry","rz","gripper"].forEach((name,i)=>{const option=node("option",name);option.value=i;options.append(option);});
    options.disabled=true;actionHost.append(actionStatus,options);$("charts").append(actionHost);
    let selectedActionEpisode=null,actionChart=null;
    function drawActions(){if(actionChart){actionChart.remove();actionChart=null;}if(!selectedActionEpisode)return;const rows=selectedActionEpisode.records.flatMap(r=>(r.output?.executed||[]).map(a=>a.applied));chart(`Applied action · ${options.selectedOptions[0].textContent}`,rows.map((row,i)=>[i,row[Number(options.value)]??null]),"Executed sample index","command (see action_spec units)");actionChart=$("charts").lastChild;}
    renderSelectedActions=episode=>{selectedActionEpisode=episode;options.disabled=!episode;actionStatus.textContent=episode?`Applied actions for rollout ${episode.rollout_index}`:"Load a run to inspect its applied actions.";drawActions();};options.addEventListener("change",drawActions);
    $("metric-table").append(detail("System performance evidence",data.performance));
    const table=node("table"),head=node("tr");["Task","Rollout","Seed","Status","Calls","Applied actions","Reason","Details"].forEach(x=>head.append(node("th",x)));table.append(head);
    data.tasks.forEach((task,taskIndex)=>task.episodes.forEach((ep,episodeIndex)=>{const tr=node("tr");[task.task_id,ep.rollout_index,ep.seed,ep.terminal_status,ep.counts.predictions,ep.counts.executed_actions,ep.terminal_reason].forEach(x=>tr.append(node("td",fmt(x))));const cell=node("td"),button=node("button","Open run");button.setAttribute("aria-label",`Open run ${ep.rollout_index}, ${task.task_id}`);button.addEventListener("click",()=>inspectEpisode(taskIndex,episodeIndex));cell.append(button);tr.append(cell);table.append(tr);}));const episodesWrap=node("div",null,"table-scroll");episodesWrap.tabIndex=0;episodesWrap.setAttribute("role","region");episodesWrap.setAttribute("aria-label","Episode outcomes");episodesWrap.append(table);$("metric-table").append(node("h3","Every finalized episode"),episodesWrap);
    const metrics=node("table"),metricHead=node("tr");["Scope","Metric","Value","Unit","Valid n","Excluded n","Unavailable n","Min","Median","Mean","Sample SD","Max","Status / reason"].forEach(x=>metricHead.append(node("th",x)));metrics.append(metricHead);
    const metricRows=(scope,rows)=>rows.forEach(metric=>{const tr=node("tr"),stats=metric.statistics||{},value=typeof metric.value==="object"&&metric.value!=null?JSON.stringify(metric.value):exactValue(metric.value);[scope,metric.metric_id,value,metric.unit,stats.valid_n??metric.sample_count,stats.excluded_n,stats.unavailable_n,stats.minimum,stats.median,stats.mean,stats.standard_deviation,stats.maximum,`${metric.status}${metric.reason?" · "+metric.reason:""}`].forEach(value=>tr.append(node("td",exactValue(value))));metrics.append(tr);});
    data.tasks.forEach(task=>{metricRows(task.task_id,task.metrics);});const metricsWrap=node("div",null,"table-scroll");metricsWrap.tabIndex=0;metricsWrap.setAttribute("role","region");metricsWrap.setAttribute("aria-label","Canonical task and episode metrics");metricsWrap.append(metrics);$("metric-table").append(node("h3","Canonical task metrics · episode metrics appear in the selected run"),metricsWrap);
    data.tasks.forEach(task=>{const {episodes,...context}=task;$("metric-table").append(detail(`Task context · ${task.task_id}`,context));});
  }
})();
