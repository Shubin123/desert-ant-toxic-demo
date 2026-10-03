import {loadToxic} from './js/infer.js';
import {contentThresholds,decide} from './js/thresholds.js';
import {LANGUAGES,guessLanguage} from './js/lang.js';
const $=id=>document.getElementById(id);
$('controls').innerHTML='<label for="input">Text to review</label><textarea id="input" maxlength="4000">Thank you for helping our neighbors. Everyone deserves respect and a safe place to live.</textarea><label for="language">Threshold language</label><select id="language"><option value="auto">Automatic (heuristic)</option></select><div class="actions"><button id="run">Analyze text</button></div><p class="hint">About 100 MB on first use. Automatic language selection is only a heuristic; override it for reliable thresholds. Long passages are truncated to the model’s recommended token limit. This is human-review triage, not an automated moderation decision or a judgment about a person. Quotation, reclamation and context can cause errors.</p>';
const names=new Intl.DisplayNames(['en'],{type:'language'});
for(const lang of LANGUAGES){const o=document.createElement('option');o.value=lang;o.textContent=names.of(lang);$('language').append(o);}
let model;
$('run').onclick=async()=>{
  $('run').disabled=true;$('output').replaceChildren();$('output').dataset.state='loading';
  try{
    const text=$('input').value.trim();if(!text)throw Error('Enter some text.');
    $('status').textContent='Loading original Toxic model...';$('progress').hidden=false;
    model??=loadToxic({onStage:s=>{$('status').textContent=`Preparing ${s}...`;},onProgress:(loaded,total)=>{$('progress').value=loaded/total;}}).catch(e=>{model=null;throw e;});
    const engine=await model;$('progress').hidden=true;$('status').textContent='Analyzing on your device...';
    const language=$('language').value==='auto'?guessLanguage(text):$('language').value;
    const result=await engine.classify(text),thresholds=contentThresholds(engine.meta,language);
    const disabled=new Set(engine.meta.disabled_heads??[]);
    const enabled=Object.fromEntries(Object.entries(result.content).filter(([label])=>!disabled.has(label)));
    const {flagged,fired}=decide(enabled,thresholds);
    const heading=document.createElement('h2');heading.textContent=flagged?'Flagged for human review':'Not flagged by these thresholds';$('output').append(heading);
    for(const [label,score] of Object.entries(result.content)){
      const row=document.createElement('p');row.textContent=`${label}: ${(score*100).toFixed(1)}% · threshold ${(thresholds[label]*100).toFixed(0)}%${disabled.has(label)?' · disabled head':''}`;$('output').append(row);
    }
    if(fired.includes('HATEFUL')){
      const targets=Object.entries(result.target).filter(([,p])=>p>=thresholds.HATEFUL).sort((a,b)=>b[1]-a[1]);
      if(targets.length){const p=document.createElement('p');p.textContent='Possible content targets: '+targets.map(([l,p])=>`${l} ${(p*100).toFixed(1)}%`).join(', ');$('output').append(p);}
    }
    const info=document.createElement('p');info.className='hint';info.textContent=`Threshold language: ${names.of(language)} · ${result.tokens} tokens · ${result.ms.toFixed(0)} ms inference. Scores describe this text, not the writer. A non-flagged result does not establish safety.`;$('output').append(info);
    $('output').dataset.state='success';$('status').textContent='Analysis complete. Review context.';
  }catch(e){$('output').textContent=e.message??String(e);$('output').dataset.state='error';$('status').textContent='Could not complete. You can retry.';}
  finally{$('progress').hidden=true;$('run').disabled=false;}
};
