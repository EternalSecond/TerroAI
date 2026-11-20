// Render local model text as DOM nodes. Model output is never interpreted as HTML.
export function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function inline(element, text) {
  const tokens = /(\x60[^\x60]+\x60|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  let offset = 0;
  for (const match of text.matchAll(tokens)) {
    element.append(text.slice(offset, match.index));
    const part = match[0];
    if (part.startsWith("[")) {
      const link = /^\[([^\]]+)\]\((.+)\)$/.exec(part);
      const anchor = node("a", link[1]);
      anchor.href = link[2];
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      element.append(anchor);
    } else if (part.startsWith("**")) {
      element.append(node("strong", part.slice(2, -2)));
    } else if (part.startsWith("*")) {
      element.append(node("em", part.slice(1, -1)));
    } else {
      element.append(node("code", part.slice(1, -1), "inline-code"));
    }
    offset = match.index + part.length;
  }
  element.append(text.slice(offset));
}

function codeBlock(text, language) {
  const frame = node("div", undefined, "code-block");
  const bar = node("div", undefined, "code-head");
  const copy = node("button", "Копіювати");
  copy.type = "button";
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Скопійовано";
    } catch {
      copy.textContent = "Не вдалося скопіювати";
    }
  };
  bar.append(node("span", language || "Код"), copy);
  const pre = node("pre");
  pre.append(node("code", text));
  frame.append(bar, pre);
  return frame;
}

function tableCells(line) {
  const cells = [""];
  let separators = 0;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "\\" && line[i + 1] === "|") {
      cells[cells.length - 1] += "|";
      i++;
    } else if (line[i] === "|") {
      cells.push("");
      separators++;
    } else cells[cells.length - 1] += line[i];
  }
  if (!separators) return [];
  if (line.trimStart().startsWith("|")) cells.shift();
  if (cells.length && !cells.at(-1).trim()) cells.pop();
  return cells.map(cell => cell.trim());
}

function tableBlock(element, lines, start) {
  const headers = tableCells(lines[start]);
  const rules = tableCells(lines[start + 1] || "");
  if (!headers.length || headers.length !== rules.length ||
      !rules.every(rule => /^:?-{3,}:?$/.test(rule))) return start;
  const table = node("table");
  const head = table.createTHead().insertRow();
  headers.forEach((value, i) => {
    const cell = node("th");
    cell.scope = "col";
    inline(cell, value);
    head.append(cell);
    if (rules[i].endsWith(":")) cell.className = rules[i].startsWith(":") ? "align-center" : "align-right";
  });
  const body = table.createTBody();
  let end = start + 2;
  for (; end < lines.length; end++) {
    const values = tableCells(lines[end]);
    if (!values.length) break;
    const row = body.insertRow();
    headers.forEach((_, i) => {
      const cell = row.insertCell();
      cell.className = head.cells[i].className;
      inline(cell, values[i] || "");
    });
  }
  const frame = node("div", undefined, "table-scroll");
  frame.tabIndex = 0;
  frame.append(table);
  element.append(frame);
  return end - 1;
}

export function markdown(element, text) {
  element.replaceChildren();
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let paragraph = null;
  let list = null;
  let fence = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (fence) {
      if (new RegExp("^\\s*" + fence.marker[0] + "{" + fence.marker.length + ",}\\s*$").test(line)) {
        element.append(codeBlock(fence.lines.join("\n"), fence.language));
        fence = null;
      } else {
        fence.lines.push(line);
      }
      continue;
    }
    const opening = /^\s*(\x60{3,}|~{3,})([\w+-]*)\s*$/.exec(line);
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    const item = /^\s*([-*+]|\d+[.)])\s+(.+)$/.exec(line);
    if (!opening && !heading && !item && line.includes("|")) {
      const end = tableBlock(element, lines, i);
      if (end > i) {
        paragraph = list = null;
        i = end;
        continue;
      }
    }
    if (opening || heading || !line.trim()) {
      paragraph = list = null;
      if (opening) fence = {marker: opening[1], language: opening[2], lines: []};
      if (heading) {
        const title = node(heading[1].length < 3 ? "h2" : "h3");
        inline(title, heading[2]);
        element.append(title);
      }
    } else if (item) {
      paragraph = null;
      const kind = /^\d/.test(item[1]) ? "OL" : "UL";
      if (!list || list.tagName !== kind) {
        list = node(kind.toLowerCase());
        element.append(list);
      }
      const li = node("li");
      inline(li, item[2]);
      list.append(li);
    } else {
      list = null;
      if (!paragraph) {
        paragraph = node("p");
        element.append(paragraph);
      } else paragraph.append(document.createElement("br"));
      inline(paragraph, line);
    }
  }
  if (fence) element.append(codeBlock(fence.lines.join("\n"), fence.language));
  return Boolean(fence);
}
