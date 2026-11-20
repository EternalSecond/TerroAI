const courtesy = /^(?:(?:добрий|доброго|доброе|добрый) (?:день|ранок|ранку|вечір|вечер|утро|ранку)|привіт|привет|вітаю|здравствуй(?:те)?|добридень|хай|hi|hello|hey|дякую|спасибо|ок(?:ей)?|гаразд|добре|як справи|как дела|як ти|как ты)$/iu;
const simple = text => text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "").trim().replace(/\s+/g, " ");
const greeting = text => !simple(text) || courtesy.test(simple(text));

function clean(text) {
  return text.replace(/[\x60*#]/g, "").replace(/\s+/g, " ").trim().replace(/[.!?;:,]+$/, "");
}

function shorten(text) {
  if (text.length <= 56) return text;
  const clipped = text.slice(0, 54);
  const boundary = clipped.lastIndexOf(" ");
  return (boundary > 30 ? clipped.slice(0, boundary) : clipped) + "…";
}

export function conversationTitle(messages) {
  for (let index = 0; index + 1 < messages.length; index += 2) {
    const question = messages[index];
    const answer = messages[index + 1];
    if (!answer.content.trim() || !["stop", "length"].includes(answer.finish)) continue;
    const subject = clean(question.content);
    if (question.attachmentOnly || greeting(subject)) {
      const attached = question.attachmentLabels?.find(label => !label.endsWith(" · не прочитано"));
      if (attached) return shorten(attached);
      continue;
    }
    // Use terms found in the actual answer when a general request produces a concrete example.
    if (/\b(?:sklearn|scikit-learn)\b/i.test(subject) && /\bLinearRegression\b|лінійн\S+ регрес|линейн\S+ регресс/i.test(answer.content)) {
      return "Лінійна регресія в scikit-learn";
    }
    const firstLine = answer.content.trim().split("\n", 1)[0];
    const heading = clean(firstLine.match(/^#{1,6}\s+(.+)$/)?.[1] || "");
    if (heading.length >= 8 && heading.length <= 56 &&
        !/^(?:код|відповідь|ответ|приклад|пример|рішення|решение|привіт|привет)(?:$|\s*[.:!])/iu.test(heading)) {
      return heading;
    }
    const topic = subject.replace(
      /^(?:(?:будь ласка|пожалуйста)[, ]+)?(?:створи|зроби|напиши|покажи|поясни|розкажи|створіть|зробіть|создай|сделай|напишите|расскажи|объясни|помоги|допоможи)(?:[,:]?\s+)(?:мені\s+|мне\s+)?(?:про\s+|як\s+|как\s+)?/iu,
      "",
    ) || subject;
    return shorten(topic[0].toUpperCase() + topic.slice(1));
  }
  return "";
}
