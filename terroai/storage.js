import {conversationTitle} from "./title.js";

const key = "terroai.workspaces.v1";

export const defaultProfile = {name: "Локальний профіль", avatar: ""};
export const isAvatar = value => typeof value === "string" && /^data:image\/webp;base64,[A-Za-z0-9+/=]+$/.test(value) &&
  value.length < 60000;

export const fresh = () => ({
  id: crypto.randomUUID(), messages: [], draft: "", effort: "low", title: "", attachments: [],
  source: "", filename: "", adapter: false, solution: "", task: null,
  sourceAttachment: "", editorOpen: false, notice: "", saved: "", running: null, scroll: null,
});

function restoreGroup(group) {
  const items = Array.isArray(group?.items) ? group.items.filter(item =>
    typeof item?.id === "string" && typeof item.draft === "string" &&
    Array.isArray(item.messages) && item.messages.every(message =>
      ["user", "assistant"].includes(message?.role) && typeof message.content === "string"),
  ).map(({detailed, ...item}) => ({
    ...fresh(), ...item, running: null,
    effort: ["low", "medium", "high"].includes(item.effort) ? item.effort : detailed ? "medium" : "low",
    attachments: recoverAttachments(item.attachments),
  })) : [];
  for (const item of items) {
    item.title ||= conversationTitle(item.messages);
    const answer = item.messages.at(-1);
    if (answer?.role !== "assistant" || answer.finish) continue;
    if (answer.content) {
      answer.finish = "stopped";
      item.solution = answer.content;
      delete item.messages.at(-2).retryAttachments;
      delete item.messages.at(-2).retryDraft;
    } else {
      item.messages.pop();
      const question = item.messages.pop();
      if (!item.draft && question?.role === "user") item.draft = question.retryDraft ?? question.content;
      item.attachments.unshift(...recoverAttachments(question?.retryAttachments));
      item.task = null;
    }
  }
  if (!items.length) items.push(fresh());
  return {items, active: items.some(item => item.id === group?.active) ? group.active : items[0].id};
}

function recoverAttachments(attachments) {
  if (!Array.isArray(attachments)) return [];
  return attachments.filter(item => typeof item?.id === "string" && typeof item.name === "string" &&
    ["text", "photo"].includes(item.kind)).map(item => ({
    ...item,
    preview: isAvatar(item.preview) ? item.preview : "",
    pending: false,
    error: item.pending ? "Обробку перервано. Додайте вкладення ще раз." : item.error,
  }));
}

export function restore() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(key));
  } catch {
    // An unavailable or invalid store must not prevent the workspace from opening.
  }
  return {
    mode: saved?.mode === "code" ? "code" : "chat",
    sidebar: saved?.sidebar !== false,
    groups: {chat: restoreGroup(saved?.groups?.chat), code: restoreGroup(saved?.groups?.code)},
    profile: {
      name: typeof saved?.profile?.name === "string" ? saved.profile.name.slice(0, 40) : defaultProfile.name,
      avatar: isAvatar(saved?.profile?.avatar) ? saved.profile.avatar : "",
    },
  };
}

export function persist({mode, sidebar, groups, profile}) {
  try {
    localStorage.setItem(key, JSON.stringify({mode, sidebar, groups, profile}, (name, value) =>
      ["running", "notice", "saved", "scroll"].includes(name) ? undefined : value));
    return true;
  } catch {
    return false;
  }
}
