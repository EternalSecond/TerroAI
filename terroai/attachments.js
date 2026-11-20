const types = {png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", bmp: "image/bmp"};
const extension = file => file.name.split(".").at(-1).toLowerCase();

export const attachmentKind = file =>
  file.type.startsWith("image/") || Object.hasOwn(types, extension(file)) ? "photo" : "text";

export async function thumbnail(file, size) {
  const bitmap = await createImageBitmap(file);
  try {
    const scale = Math.min(size / bitmap.width, size / bitmap.height, 1);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/webp", .8);
  } finally {
    bitmap.close();
  }
}

export async function readPhoto(file) {
  const mime = Object.values(types).includes(file.type) ? file.type :
    (Object.hasOwn(types, extension(file)) ? types[extension(file)] : undefined);
  if (!mime) throw new Error("Підтримуються фото у форматах PNG, JPEG і BMP.");
  // Same-origin endpoint. The model request later includes only recognized text.
  const response = await fetch("api/ocr", {
    method: "POST", headers: {"Content-Type": mime}, body: file,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message || "Не вдалося прочитати фото.");
  return result;
}

export async function readText(file) {
  try {
    const text = new TextDecoder("utf-8", {fatal: true}).decode(await file.arrayBuffer());
    if (text.includes("\0")) throw new Error("Файл має бути текстовим у UTF-8.");
    if (!text.trim()) throw new Error("Текстовий файл порожній.");
    return text;
  } catch (error) {
    if (error instanceof TypeError) throw new Error("Файл має бути текстовим у UTF-8.");
    throw error;
  }
}
