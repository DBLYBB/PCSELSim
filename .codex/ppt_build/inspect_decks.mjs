import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "file:///C:/Users/19254/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const [outputDir, ...inputs] = process.argv.slice(2);
await fs.mkdir(outputDir, { recursive: true });

for (const input of inputs) {
  const deck = await PresentationFile.importPptx(await FileBlob.load(input));
  const base = path.basename(input, path.extname(input));
  const snapshot = await deck.inspect({
    kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
    include: "id,slide,name,title,text,textPreview,textChars,textLines,bbox,bboxUnit,isPlaceholder,alt,rows,cols,chartType,placeholders",
    maxChars: 50000,
  });
  await fs.writeFile(path.join(outputDir, `${base}.inspect.ndjson`), snapshot.ndjson, "utf8");
  const montage = await deck.export({ format: "png", montage: true, scale: 0.55 });
  await fs.writeFile(path.join(outputDir, `${base}.montage.png`), new Uint8Array(await montage.arrayBuffer()));
  const layouts = [];
  for (let i = 0; i < deck.slides.items.length; i += 1) {
    const slide = deck.slides.items[i];
    const png = await slide.export({ format: "png", scale: 1 });
    await fs.writeFile(path.join(outputDir, `${base}.slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    layouts.push(await layout.text());
  }
  await fs.writeFile(path.join(outputDir, `${base}.layouts.jsonl`), layouts.join("\n"), "utf8");
  console.log(JSON.stringify({ input, slides: deck.slides.items.length, masters: deck.masters.items.length, layouts: deck.layouts.items.length }));
}
