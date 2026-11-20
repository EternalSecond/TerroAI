# Сторонні компоненти та ліцензії

## Базова модель

[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) завантажується окремо з Hugging Face. Перевірена ревізія `b968826d9c46dd6066d109eabc6255188de91218` має [ліцензію Apache 2.0](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/LICENSE). У релізі TerroAI немає ваг, токенізатора, шаблону моделі або адаптерів. Якщо їх додадуть до окремого дистрибутива, потрібно виконати умови розділу 4 ліцензії щодо її копії, позначення змін та збереження застосовних авторських повідомлень і NOTICE.

## Залежності

Нижче наведено основні ліцензії прямих залежностей за метаданими й файлами ліцензій версій з `requirements.txt`. Пакети встановлюються окремо. Віртуальне середовище, CUDA-бінарники й сторонні бібліотеки не входять до дерева релізу.

| Пакети | Основна ліцензія |
| --- | --- |
| PyTorch, Uvicorn | BSD-3-Clause |
| Transformers, Accelerate, PEFT, TRL, Datasets, Hugging Face Hub, hf-xet, Safetensors | Apache-2.0 |
| bitsandbytes, FastAPI, Pydantic | MIT |
| filelock | Unlicense |

[PyTorch 2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) має окремі повідомлення про включені компоненти. Для перевірки повного переліку підпорядкованих ліцензій пакетів використовуйте файли LICENSE і NOTICE з установлених дистрибутивів. Скрипт OCR викликає API Windows і не містить бінарників або мовних пакетів Microsoft. Вебінтерфейс не постачає сторонні JS-бібліотеки чи шрифти.

## Код і матеріали TerroAI

Ліцензію на код, синтетичні приклади та скріншоти TerroAI ще не обрано. Apache 2.0 моделі не надає прав на ці файли. Перед публікацією власник має підтвердити бажані умови та додати відповідний `LICENSE`, якщо хоче дозволити повторне використання. Див. [документацію GitHub про ліцензування](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
