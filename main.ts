import { GoogleGenAI } from "@google/genai";
import type { GenerateContentConfig, Part } from "@google/genai";
import { Buffer } from "node:buffer";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { config } from './config.js';

const API_KEY = 'API_KEY';
const REMOTE_LINK_REGEX = /\b(?:https?:\/\/|file:\/\/)[^\s<>"')]+/gi;
const SUPPORTED_FILE_TYPES: Record<string, string> = {
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.heic': 'image/heic',
  '.heif': 'image/heif',
};

interface Completion {
  Content: string;
  TokenUsage?: number;
}

interface ErrorCompletion {
  Error: string;
}

type ConnectorResponse = {
  Completions: Array<Completion | ErrorCompletion>;
  ModelType?: string;
};

const mapErrorToCompletion = (error: unknown): ErrorCompletion => {
  const errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
  return {
    Error: errorMessage,
  };
};

const stripWrappingCharacters = (value: string): string => {
  let candidate = value.trim().replace(/^[<({\['"]+/, '');
  while (candidate && /[>)}\]'".,;!]+$/.test(candidate)) {
    candidate = candidate.slice(0, -1);
  }
  return candidate;
};

const getExtension = (input: string): string | null => {
  const base = input.split('?')[0]?.split('#')[0] ?? input;
  const ext = path.extname(base);
  return ext ? ext.toLowerCase() : null;
};

const isHttpUrl = (value: string): boolean => {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
};

const isFileUrl = (value: string): boolean => {
  try {
    const url = new URL(value);
    return url.protocol === 'file:';
  } catch {
    return false;
  }
};

const resolveLocalPath = (reference: string): string => {
  if (isFileUrl(reference)) {
    return fileURLToPath(new URL(reference));
  }

  if (reference.startsWith('~')) {
    return path.join(os.homedir(), reference.slice(1));
  }

  return path.isAbsolute(reference) ? reference : path.resolve(reference);
};

const extractFileReferences = (text: string): string[] => {
  const references = new Set<string>();

  const remoteMatches = text.match(REMOTE_LINK_REGEX) ?? [];
  for (const rawMatch of remoteMatches) {
    const candidate = stripWrappingCharacters(rawMatch);
    const ext = getExtension(candidate.startsWith('file://') ? fileURLToPath(new URL(candidate)) : candidate);
    if (ext && SUPPORTED_FILE_TYPES[ext]) {
      references.add(candidate);
    }
  }

  for (const rawToken of text.split(/\s+/)) {
    const candidate = stripWrappingCharacters(rawToken);
    if (!candidate || isHttpUrl(candidate) || isFileUrl(candidate)) {
      continue;
    }
    const ext = getExtension(candidate);
    if (ext && SUPPORTED_FILE_TYPES[ext]) {
      references.add(candidate);
    }
  }

  return Array.from(references);
};

const guessMimeType = (reference: string, fallbackExt: string): string => {
  const ext = getExtension(reference);
  if (ext && SUPPORTED_FILE_TYPES[ext]) {
    return SUPPORTED_FILE_TYPES[ext];
  }
  return SUPPORTED_FILE_TYPES[fallbackExt] ?? 'application/octet-stream';
};

const loadInlineDataPart = async (reference: string): Promise<Part | null> => {
  const resolvedRef = stripWrappingCharacters(reference);
  const ext = getExtension(resolvedRef.startsWith('file://') ? fileURLToPath(new URL(resolvedRef)) : resolvedRef);

  if (!ext || !SUPPORTED_FILE_TYPES[ext]) {
    return null;
  }

  try {
    if (isHttpUrl(resolvedRef)) {
      const response = await fetch(resolvedRef);
      if (!response.ok) {
        console.warn(`Failed to fetch remote file: ${resolvedRef} (${response.status})`);
        return null;
      }
      const arrayBuffer = await response.arrayBuffer();
      const base64 = Buffer.from(arrayBuffer).toString('base64');
      const headerMime = response.headers.get('content-type')?.split(';')[0]?.trim();
      return {
        inlineData: {
          mimeType: headerMime && SUPPORTED_FILE_TYPES[ext] !== headerMime ? headerMime : SUPPORTED_FILE_TYPES[ext],
          data: base64,
        },
      };
    }

    const filePath = resolveLocalPath(resolvedRef);
    const data = await fs.readFile(filePath);
    return {
      inlineData: {
        mimeType: guessMimeType(filePath, ext),
        data: data.toString('base64'),
      },
    };
  } catch (error) {
    console.warn(`Unable to load file reference "${resolvedRef}":`, error);
    return null;
  }
};

const buildMessageParts = async (prompt: string): Promise<Array<string | Part>> => {
  const references = extractFileReferences(prompt);
  if (references.length === 0) {
    return [{ text: prompt }];
  }

  const inlineParts: Part[] = [];
  for (const reference of references) {
    const part = await loadInlineDataPart(reference);
    if (part) {
      inlineParts.push(part);
    }
  }

  return [{ text: prompt }, ...inlineParts];
};

async function main(
  model: string,
  prompts: string[],
  properties: Record<string, unknown>,
  settings: Record<string, unknown>,
): Promise<ConnectorResponse> {
  try {
    const { ...restProperties } = properties;

    const rawApiKey = settings?.[API_KEY];
    const apiKey = typeof rawApiKey === 'string' && rawApiKey.trim().length > 0 ? rawApiKey.trim() : undefined;
    const ai = new GoogleGenAI(apiKey ? { apiKey } : {});
    const chatConfig =
      Object.keys(restProperties).length > 0 ? (restProperties as GenerateContentConfig) : undefined;

    const chat = ai.chats.create({
      model,
      ...(chatConfig ? { config: chatConfig } : {}),
    });

    const outputs: Array<Completion | ErrorCompletion> = [];

    for (const prompt of prompts) {
      try {
        const messageParts = await buildMessageParts(prompt);
        const result = await chat.sendMessage({ message: messageParts });
        const text = result.text ?? '';

        // Count tokens
        const { totalTokens } = await ai.models.countTokens({
          model,
          contents: messageParts,
        });

        outputs.push({ Content: text, TokenUsage: totalTokens });
      } catch (error) {
        const completionWithError = mapErrorToCompletion(error);
        outputs.push(completionWithError);
      }
    }

    return {
      Completions: outputs,
    };
  } catch (error) {
    console.error('Error in main function:', error);
    return {
      Completions: [mapErrorToCompletion(error)],
    };
  }
}

export { main, config };
