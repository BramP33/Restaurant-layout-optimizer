// Dunne laag over de server-API.

async function json(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Serverfout ${res.status}`);
  return body;
}

export const api = {
  health: () => fetch("/api/health").then(json),
  preview: (project) => fetch("/api/preview", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project }),
  }).then(json),
  createJob: (project, quality, top) => fetch("/api/jobs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project, quality, top }),
  }).then(json),
  job: (id) => fetch(`/api/jobs/${id}`).then(json),
  cancel: (id) => fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(json),
};

// Het project zonder foto en zonder resultaten: dat is alles wat de server
// nodig heeft, en het houdt de aanvraag klein.
export function projectForServer(p) {
  const { photo, result, job, view, ...rest } = p;
  return rest;
}
