import{e as h,r as x,j as e,B as o,ao as l}from"./index-ck01-P-l.js";/**
 * @license lucide-react v0.319.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const n=h("Clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]),a=window.location.origin,d=`<iframe 
src="${a}/open-kf-chatbot"
title="Chatbot"
style="min-width: 420px;min-height: 60vh"
frameborder="0"
></iframe>`,m=`<script 
src="${a}/open-kf-chatbot/embed.js" 
bot-domain="${a}/open-kf-chatbot" 
defer
><\/script>`,f=()=>{const[i,r]=x.useState({script:!1,iframe:!1}),c=t=>{navigator.clipboard.writeText(t==="script"?m:d),r(s=>({...s,[t]:!0})),setTimeout(()=>{r(s=>({...s,[t]:!1}))},1500)};return e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"mt-2",children:e.jsx("p",{className:"text-sm text-zinc-500",children:"To add a chat bubble to the bottom right of your website add this script tag to your html"})}),e.jsx("div",{className:"mt-5",children:e.jsx("pre",{className:"w-fit overflow-auto rounded bg-zinc-100 p-2 text-xs",children:e.jsx("code",{children:m})})}),e.jsxs(o,{className:"mt-6",variant:"outline",onClick:()=>c("script"),children:["Copy Script",i.script?e.jsx(l,{className:"h-4 w-4 ml-2"}):e.jsx(n,{className:"h-4 w-4 ml-2"})]}),e.jsx("div",{className:"mt-2",children:e.jsx("p",{className:"text-sm text-zinc-500",children:"To add the chatbot any where on your website, add this iframe to your html code"})}),e.jsx("div",{className:"mt-5",children:e.jsx("pre",{className:"w-fit overflow-auto rounded bg-zinc-100 p-2 text-xs",children:e.jsx("code",{children:d})})}),e.jsxs(o,{className:"mt-6",variant:"outline",onClick:()=>c("iframe"),children:["Copy Iframe",i.iframe?e.jsx(l,{className:"h-4 w-4 ml-2"}):e.jsx(n,{className:"h-4 w-4 ml-2"})]})]})};export{f as Embed};
