import {markdown, node} from "./markdown.js";
import {defaultProfile, fresh, isAvatar, persist, restore} from "./storage.js";
import {attachmentKind, readPhoto, readText, thumbnail} from "./attachments.js";
import {conversationTitle} from "./title.js";

const $ = id => document.getElementById(id);
const workspace = restore();
let mode = workspace.mode;
let settings = {adapter: false, file_limit: 524288, photo_limit: 8 * 1024 * 1024};
const effortLabels = {low: "Низьке", medium: "Середнє", high: "Високе"};
const group = () => workspace.groups[mode];
const state = () => group().items.find(item => item.id === group().active);
const scroll = $("scroll");
const timeline = $("timeline");
const wide = window.matchMedia("(min-width: 761px)");
let mobileSidebar = false;
let saveTimer;

function saveState() {
  clearTimeout(saveTimer);
  saveTimer = undefined;
  $("storage-notice").hidden = persist(workspace);
  return $("storage-notice").hidden;
}

function checkpoint() {
  saveTimer ??= setTimeout(saveState, 700);
}

window.addEventListener("pagehide", saveState);

function showSidebar() {
  const visible = wide.matches ? workspace.sidebar : mobileSidebar;
  $("sidebar").hidden = !visible;
  $("sidebar-toggle").setAttribute("aria-expanded", String(visible));
  $("sidebar-backdrop").hidden = wide.matches || !visible;
}

wide.addEventListener("change", () => { mobileSidebar = false; showSidebar(); });

function renderHistory() {
  const active = state();
  const buttons = [...group().items].reverse().map(item => {
    const title = item.title || (mode === "code" ? "Нова задача" : "Новий чат");
    const button = node("button", title);
    button.type = "button";
    button.title = title.slice(0, 200);
    button.setAttribute("aria-current", String(item === active));
    button.onclick = () => {
      active.scroll = scroll.scrollTop;
      group().active = item.id;
      mobileSidebar = false;
      showMode(mode);
    };
    const row = node("li");
    row.append(button);
    return row;
  });
  $("history").replaceChildren(...buttons);
}

function showEditor() {
  const visible = mode === "code" && state().editorOpen;
  $("workbench").hidden = !visible;
  $("workspace").classList.toggle("editor-open", visible);
  $("editor-toggle").setAttribute("aria-expanded", String(visible));
  $("editor-toggle").textContent = visible ? "Згорнути редактор" : "Відкрити редактор";
}

function toggleEditor(visible) {
  state().editorOpen = visible;
  showEditor();
  saveState();
}

function controls() {
  const session = state();
  $("notice").textContent = session.notice;
  $("stop").hidden = !session.running;
  $("send").disabled = !canSend(session);
  $("solution").readOnly = Boolean(session.running);
  $("save").disabled = Boolean(session.running) || !session.task || !session.solution.trim();
  $("save-status").textContent = session.saved;
}

function canSend(session) {
  return !session.running && !session.attachments.some(item => item.pending) &&
    Boolean(session.draft.trim() || session.attachments.some(item => item.text?.trim()));
}

function renderAttachments(session) {
  $("attachments").replaceChildren(...session.attachments.map(item => {
    const chip = node("li", undefined, "attachment");
    if (item.preview) {
      const preview = node("img");
      preview.src = item.preview;
      preview.alt = "";
      chip.append(preview);
    }
    const label = node("span", item.name, "attachment-name");
    label.title = item.name;
    const status = item.pending ? (item.kind === "photo" ? "Розпізнаю текст…" : "Читаю файл…") :
      item.error || (item.kind === "photo" ? "Текст розпізнано" : "");
    if (status) {
      const detail = node("small", status, item.error ? "attachment-error" : "");
      detail.setAttribute("role", "status");
      label.append(detail);
    }
    const remove = node("button", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", "Видалити вкладення " + item.name);
    remove.onclick = () => {
      session.attachments = session.attachments.filter(other => other !== item);
      if (session.sourceAttachment === item.id) {
        session.source = session.filename = session.sourceAttachment = "";
        $("source").value = "";
      }
      renderAttachments(session);
      controls();
      saveState();
      $("add-menu").querySelector("summary").focus();
    };
    chip.append(label, remove);
    return chip;
  }));
}

function reply(article, message, index, session) {
  const content = node("div");
  const openFence = markdown(content, message.content);
  article.replaceChildren(node("span", "TerroAI", "reply-label"), content);
  const latest = index === session.messages.length - 1;
  if (latest && !session.running && (openFence || ["length", "stopped", "error"].includes(message.finish))) {
    const actions = node("div", undefined, "reply-actions");
    const status = {length: "Досягнуто ліміту відповіді.", stopped: "Відповідь зупинено.", error: "Відповідь перервано."};
    const more = node("button", "Продовжити");
    more.type = "button";
    more.onclick = () => generate(true);
    actions.append(node("span", status[message.finish] || "Блок коду ще відкритий."), more);
    article.append(actions);
  }
}

function render(session, position = null) {
  if (session !== state()) return;
  $("workspace").classList.toggle("start", !session.messages.length);
  timeline.replaceChildren();
  if (!session.messages.length) {
    const empty = node("div", undefined, "empty");
    empty.append(
      node("p", mode === "code" ? "TerroCreate" : "TerroMind", "mode-label"),
      node("h1", mode === "code" ? "Опишіть задачу" : "Чим допомогти?"),
    );
    timeline.append(empty);
  }
  session.messages.forEach((message, index) => {
    const article = node("article", undefined, "message " + message.role);
    if (message.role === "user") {
      article.textContent = message.content;
      const labels = message.attachmentLabels || (message.sourceLabel ? [message.sourceLabel] : []);
      labels.forEach(label => article.append(node("small", label, "file-tag")));
    } else reply(article, message, index, session);
    timeline.append(article);
  });
  controls();
  renderAttachments(session);
  renderHistory();
  scroll.scrollTop = position ?? scroll.scrollHeight;
}

function fitDraft() {
  const draft = $("draft");
  draft.style.height = "auto";
  draft.style.height = draft.scrollHeight + "px";
}

function showMode(next) {
  if (next !== mode) state().scroll = scroll.scrollTop;
  mode = next;
  workspace.mode = next;
  const session = state();
  $("workspace").classList.toggle("code", mode === "code");
  $("history-label").textContent = mode === "code" ? "Задачі" : "Чати";
  $("reset").textContent = mode === "code" ? "Нова задача" : "Новий чат";
  document.querySelectorAll(".code-only").forEach(element => { element.hidden = mode !== "code"; });
  document.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-current", String(button.dataset.mode === mode)));
  $("draft").placeholder = mode === "code" ? "Наприклад, створіть приклад лінійної регресії…" : "Напишіть повідомлення…";
  $("draft").value = session.draft;
  $("effort-label").textContent = effortLabels[session.effort];
  document.querySelectorAll("[name=effort]").forEach(radio => { radio.checked = radio.value === session.effort; });
  ["add-menu", "effort-menu"].forEach(id => { $(id).open = false; });
  $("source").value = session.source;
  $("solution").value = session.solution;
  $("adapter").checked = session.adapter;
  showSidebar();
  showEditor();
  fitDraft();
  render(session, session.scroll);
  saveState();
}

function bodyMessages(session, continuation) {
  // Keep the complete transcript in the browser. Send a bounded suffix of full turns.
  const history = continuation ? session.messages.slice(-80) : session.messages.slice(0, -1).slice(-81);
  return history.map(message => ({
    role: message.role, content: message.prompt || message.content,
  }));
}

async function readResponse(response, onItem) {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error?.message || "Не вдалося виконати запит.");
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += value;
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const line = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (line === "data: [DONE]") return;
        if (!line.startsWith("data: ")) continue;
        const item = JSON.parse(line.slice(6));
        if (item.error) throw new Error(item.error.message);
        onItem(item.choices?.[0]);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}

async function generate(continuation = false) {
  const session = state();
  const requestMode = mode;
  if (session.running || (!continuation && !canSend(session))) return;
  const request = new AbortController();
  session.running = request;
  session.notice = !continuation && session.effort === "high" ? "Готую відповідь із високим зусиллям…" : "Готую відповідь…";
  session.saved = "";
  let response;
  if (continuation) {
    response = session.messages.at(-1);
    response.finish = null;
  } else {
    const queued = session.attachments;
    const task = session.draft.trim() || "Поясніть вміст вкладень.";
    const manualSource = requestMode === "code" && !session.sourceAttachment && session.source.trim() ? session.source : "";
    const context = queued.flatMap(item => item.text ? [
      (item.kind === "photo" ? "Текст, розпізнаний на фото " : "Вміст файлу ") +
      JSON.stringify(item.name) + ":\n" + item.text,
    ] : []);
    if (manualSource) context.unshift("Початковий код:\n" + manualSource);
    const source = context.join("\n\n");
    const labels = queued.map(item => item.name + (item.error ? " · не прочитано" : ""));
    if (manualSource) labels.unshift(session.filename || "Початковий код");
    const user = {
      role: "user", content: task, attachmentLabels: labels, retryAttachments: queued, retryDraft: session.draft,
      attachmentOnly: !session.draft.trim(),
    };
    if (source) user.prompt = task + "\n\n" + source;
    session.attachments = [];
    session.messages.push(user);
    response = {role: "assistant", content: ""};
    session.messages.push(response);
    let datasetSource = manualSource;
    if (queued.length === 1 && queued[0].kind === "text" && !manualSource) datasetSource = queued[0].text || "";
    else if (queued.length) datasetSource = source;
    session.task = {task, source: datasetSource};
    session.draft = "";
    $("draft").value = "";
    fitDraft();
  }
  render(session);
  saveState();
  let frame = 0;
  const paint = () => {
    if (session !== state() || frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (session !== state() || session.running !== request) return;
      const pinned = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
      reply(timeline.lastElementChild, response, session.messages.length - 1, session);
      controls();
      if (pinned) scroll.scrollTop = scroll.scrollHeight;
    });
  };
  try {
    const result = await fetch("api/generate", {
      method: "POST", headers: {"Content-Type": "application/json"}, signal: request.signal,
      body: JSON.stringify({
        mode: requestMode, messages: bodyMessages(session, continuation),
        adapter: session.adapter, effort: session.effort, continuation,
      }),
    });
    await readResponse(result, choice => {
      if (!choice) return;
      if (choice.delta?.content) {
        response.content += choice.delta.content;
        checkpoint();
        session.notice = "";
        paint();
      }
      if (choice.finish_reason) response.finish = choice.finish_reason;
    });
    if (!response.finish) throw new Error("Потік відповіді перервано.");
    session.notice = "";
  } catch (error) {
    response.finish = request.signal.aborted ? "stopped" : "error";
    session.notice = request.signal.aborted ? "Генерацію зупинено." : error.message;
  } finally {
    if (frame) cancelAnimationFrame(frame);
    session.running = null;
    if (!response.content.trim()) {
      session.messages.pop();
      const unanswered = session.messages.pop();
      if (!session.draft) session.draft = unanswered?.retryDraft ?? unanswered?.content ?? "";
      session.attachments.unshift(...(unanswered?.retryAttachments || []));
      if (response.finish === "length") session.notice = "Ліміт вичерпано до появи відповіді. Спробуйте нижче зусилля або коротший запит.";
      else if (response.finish === "stop" && !session.notice) session.notice = "Модель не повернула текст відповіді.";
      session.task = null;
      if (session === state()) {
        $("draft").value = session.draft;
        fitDraft();
      }
    } else {
      const question = session.messages.at(-2);
      delete question.retryAttachments;
      delete question.retryDraft;
      session.title ||= conversationTitle(session.messages);
      session.solution = response.content;
      if (session === state()) $("solution").value = session.solution;
    }
    const pinned = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
    render(session, pinned ? null : scroll.scrollTop);
    saveState();
  }
}

document.querySelectorAll("[data-mode]").forEach(button => button.onclick = () => {
  mobileSidebar = false;
  showMode(button.dataset.mode);
});
$("sidebar-toggle").onclick = () => {
  if (wide.matches) workspace.sidebar = !workspace.sidebar;
  else mobileSidebar = !mobileSidebar;
  showSidebar();
  saveState();
};
$("sidebar-backdrop").onclick = () => { mobileSidebar = false; showSidebar(); };
$("editor-toggle").onclick = () => toggleEditor(!state().editorOpen);
$("close-editor").onclick = () => { toggleEditor(false); $("editor-toggle").focus(); };
$("composer").onsubmit = event => { event.preventDefault(); generate(); };
$("draft").oninput = event => { state().draft = event.target.value; fitDraft(); controls(); saveState(); };
$("draft").onkeydown = event => {
  if (event.key !== "Enter" || event.isComposing) return;
  event.preventDefault();
  if (event.shiftKey) {
    const draft = event.target;
    draft.setRangeText("\n", draft.selectionStart, draft.selectionEnd, "end");
    state().draft = draft.value;
    fitDraft();
    controls();
    saveState();
  } else generate();
};
$("stop").onclick = () => state().running?.abort();
$("reset").onclick = () => {
  const old = state();
  old.scroll = scroll.scrollTop;
  const next = {...fresh(), adapter: old.adapter};
  group().items.push(next);
  group().active = next.id;
  mobileSidebar = false;
  showMode(mode);
  $("draft").focus();
};
function closeEffort() {
  $("effort-menu").open = false;
  $("effort-menu").querySelector("summary").focus();
}

document.querySelectorAll("[name=effort]").forEach(radio => {
  radio.onchange = () => {
    state().effort = radio.value;
    $("effort-label").textContent = effortLabels[radio.value];
    saveState();
  };
  radio.onclick = event => { if (event.detail > 0) closeEffort(); };
  radio.onkeydown = event => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    closeEffort();
  };
});

const menus = ["add-menu", "effort-menu"].map($);
document.addEventListener("click", event => {
  menus.forEach(menu => { if (!menu.contains(event.target)) menu.open = false; });
});
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  menus.forEach(menu => {
    if (!menu.open) return;
    menu.open = false;
    menu.querySelector("summary").focus();
  });
});

$("adapter").onchange = event => { state().adapter = event.target.checked; saveState(); };
$("source").oninput = event => {
  const session = state();
  session.source = event.target.value;
  const attachment = session.attachments.find(item => item.id === session.sourceAttachment);
  if (attachment) {
    attachment.text = session.source;
  } else {
    session.filename = session.sourceAttachment = "";
  }
  controls();
  saveState();
};
$("solution").oninput = event => { state().solution = event.target.value; controls(); saveState(); };
function attachFiles(files, kind) {
  const session = state();
  const requestMode = mode;
  const jobs = [];
  const errors = new Set();
  for (const file of files) {
    const type = kind || attachmentKind(file);
    if (session.attachments.length >= 4) {
      errors.add("Можна додати до чотирьох вкладень.");
      break;
    }
    if (file.size > (type === "photo" ? settings.photo_limit : settings.file_limit)) {
      errors.add(type === "photo" ? "Фото завелике. Максимум 8 МіБ." :
        "Файл завеликий. Вставте потрібний фрагмент.");
      continue;
    }
    const item = {id: crypto.randomUUID(), kind: type, name: file.name, pending: true};
    session.attachments.push(item);
    jobs.push({file, item});
  }
  session.notice = [...errors].join(" ");
  renderAttachments(session);
  controls();
  saveState();
  const sourceIsFromPastTurn = session.sourceAttachment &&
    !session.attachments.some(item => item.id === session.sourceAttachment);
  const editorAvailable = !session.source.trim() || sourceIsFromPastTurn;
  const editorItem = requestMode === "code" && editorAvailable ?
    jobs.find(({item}) => item.kind === "text")?.item : null;
  for (const {file, item} of jobs) readAttachment(session, file, item, item === editorItem);
}

async function readAttachment(session, file, item, openInEditor) {
  const photo = item.kind === "photo";
  const previousSource = session.source;
  const update = () => {
    if (!session.attachments.includes(item)) return;
    if (session === state()) { renderAttachments(session); controls(); }
    saveState();
  };
  if (photo) {
    thumbnail(file, 104).then(preview => { item.preview = preview; update(); }).catch(() => {});
  }
  try {
    item.text = photo ? (await readPhoto(file)).text : await readText(file);
    if (openInEditor && session.source === previousSource && session.attachments.includes(item)) {
      session.source = item.text;
      session.filename = item.name;
      session.sourceAttachment = item.id;
      session.editorOpen = true;
      if (session === state()) {
        $("source").value = item.text;
        showEditor();
      }
    }
  } catch (error) {
    item.error = error.message;
  } finally {
    item.pending = false;
    update();
  }
}

for (const [id, kind] of [["file", "text"], ["photo", "photo"]]) {
  $("choose-" + id).onclick = () => {
    $("add-menu").open = false;
    $("add-menu").querySelector("summary").focus();
    $(id).click();
  };
  $(id).onchange = event => {
    const files = Array.from(event.target.files);
    event.target.value = "";
    if (files.length) attachFiles(files, kind);
  };
}

const dropZone = document.querySelector(".conversation");
const fileDrag = event => Array.from(event.dataTransfer?.types || []).includes("Files");
function showDropHint(visible) {
  dropZone.classList.toggle("file-drag", visible);
  $("drop-hint").hidden = !visible;
}
// Cancel file navigation even when the drop misses the conversation. Text
// and link drags keep their browser behavior.
window.addEventListener("dragover", event => {
  if (!fileDrag(event)) return;
  event.preventDefault();
  const inside = dropZone.contains(event.target);
  event.dataTransfer.dropEffect = inside ? "copy" : "none";
  showDropHint(inside);
});
dropZone.addEventListener("dragenter", event => {
  if (!fileDrag(event)) return;
  event.preventDefault();
  showDropHint(true);
});
dropZone.addEventListener("dragleave", event => {
  if (!dropZone.contains(event.relatedTarget)) showDropHint(false);
});
window.addEventListener("drop", event => {
  showDropHint(false);
  if (!fileDrag(event)) return;
  event.preventDefault();
  if (dropZone.contains(event.target) && event.dataTransfer.files.length) {
    attachFiles(Array.from(event.dataTransfer.files));
  }
});
window.addEventListener("dragend", () => showDropHint(false));
$("save").onclick = async () => {
  const session = state();
  try {
    const response = await fetch("api/dataset", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...session.task, solution: session.solution}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || "Не вдалося зберегти.");
    session.saved = result.added ? "Збережено в датасеті." : "Така пара вже є в датасеті.";
  } catch (error) {
    session.saved = error.message;
  }
  if (session === state()) controls();
};

function paintAvatar(element, profile) {
  if (profile.avatar) {
    const image = node("img");
    image.src = profile.avatar;
    image.alt = "";
    element.replaceChildren(image);
  } else {
    element.textContent = Array.from(profile.name.trim() || defaultProfile.name)[0].toUpperCase();
  }
}

function paintProfile() {
  $("profile-name").textContent = workspace.profile.name;
  paintAvatar($("profile-avatar"), workspace.profile);
}

const dialog = $("settings");
let profileDraft;
function previewProfile() {
  paintAvatar($("avatar-preview"), profileDraft);
  $("clear-avatar").hidden = !profileDraft.avatar;
}

$("profile").onclick = () => {
  profileDraft = {...workspace.profile};
  $("display-name").value = profileDraft.name;
  $("profile-status").textContent = "";
  $("profile-submit").disabled = false;
  previewProfile();
  dialog.showModal();
};
$("close-settings").onclick = () => dialog.close();
$("display-name").oninput = event => { profileDraft.name = event.target.value; previewProfile(); };
$("choose-avatar").onclick = () => $("avatar-file").click();
$("clear-avatar").onclick = () => { profileDraft.avatar = ""; previewProfile(); $("choose-avatar").focus(); };
$("avatar-file").onchange = async event => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  const target = profileDraft;
  const status = $("profile-status");
  status.textContent = "";
  $("profile-submit").disabled = true;
  try {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || file.size > settings.photo_limit) {
      throw new Error("Виберіть PNG, JPEG або WebP до 8 МіБ.");
    }
    const avatar = await thumbnail(file, 128);
    if (!isAvatar(avatar)) throw new Error("Не вдалося підготувати аватар.");
    if (dialog.open && profileDraft === target) {
      target.avatar = avatar;
      previewProfile();
    }
  } catch (error) {
    if (dialog.open && profileDraft === target) status.textContent = error.message;
  } finally {
    if (dialog.open && profileDraft === target) $("profile-submit").disabled = false;
  }
};
$("profile-form").onsubmit = event => {
  event.preventDefault();
  const previous = workspace.profile;
  workspace.profile = {name: profileDraft.name.trim() || defaultProfile.name, avatar: profileDraft.avatar};
  if (!saveState()) {
    workspace.profile = previous;
    $("profile-status").textContent = "Браузер не зміг зберегти профіль.";
    return;
  }
  paintProfile();
  dialog.close();
};

paintProfile();
showMode(mode);
fetch("api/settings").then(response => response.json()).then(data => {
  settings = {...settings, ...data};
  $("model").textContent = settings.model;
  $("adapter").disabled = !settings.adapter;
  if (!settings.adapter) $("adapter").parentElement.title = "Локальний адаптер ще не створено";
}).catch(() => { $("model").textContent = "Сервер недоступний"; });
