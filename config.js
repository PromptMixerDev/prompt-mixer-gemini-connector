export const config = {
  connectorName: "Gemini",
  models: [
    "gemini-1.0-pro",
    "gemini-1.5-pro", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro-exp-0801", 
    "gemini-2.0-flash", 
    "gemini-2.0-flash-lite-preview-02-05", 
    "gemini-1.5-flash-8b",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-exp-1206",
    "learnlm-1.5-pro-experimental"
  ],
  properties: [
    {
      id: "maxOutputTokens",
      name: "Max Output Tokens",
      value: 300,
      type: "number",
    },
    {
      id: "stopSequences",
      name: "Stop Sequences",
      value: ["red"], // Array of strings
      type: "array",
    },
    {
      id: "temperature",
      name: "Temperature",
      value: 0.9, // Float
      type: "number",
    },
    {
      id: "topP",
      name: "Top P",
      value: 0.1, // Float
      type: "number",
    },
    {
      id: "topK",
      name: "Top K",
      value: 16, // Integer
      type: "number",
    },
  ],

  settings: [{ id: "API_KEY", name: "API Key", value: "", type: "string" }],
  author: "Prompt Mixer",
  description: "Gemini API connector",
  iconBase64:
    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMTQuNUM3LjQ2MTQ1IDExLjE0MzMgNC42NzE4MyA4LjUwODYzIDEuMTE3NjUgOEM0LjY3MTgzIDcuNDkxMzcgNy40NjE0NSA0Ljg1NjczIDggMS41QzguNTM4NTUgNC44NTY3MyAxMS4zMjgyIDcuNDkxMzcgMTQuODgyNCA4QzExLjMyODQgOC41MDg2MyA4LjUzODc2IDExLjE0MzMgOCAxNC41WiIgZmlsbD0iIzZGNzM3QSIvPgo8L3N2Zz4K",
};
