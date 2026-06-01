import { createHash } from "node:crypto";
import { CONFIG, XMEM_USERNAME } from "../config.js";

function sha256(input: string): string {
  return createHash("sha256").update(input).digest("hex").slice(0, 16);
}

export function getUserId(): string {
  return XMEM_USERNAME || "anonymous";
}

export function getProjectUserId(directory: string): string {
  const username = XMEM_USERNAME || "anonymous";
  return `${username}_project_${sha256(directory)}`;
}

export function getUserIds(directory: string): { user: string; project: string } {
  return {
    user: getUserId(),
    project: getProjectUserId(directory),
  };
}

export function resolveUserId(scope: "user" | "project", directory: string): string {
  const ids = getUserIds(directory);
  return scope === "user" ? ids.user : ids.project;
}

export function getTags(directory: string): { user: string; project: string } {
  return getUserIds(directory);
}

export { CONFIG };
