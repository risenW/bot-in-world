// Prefix a root-relative asset path with Vite's base URL so bundled assets
// resolve under a GitHub Pages project subpath (e.g. /spaitial-bot/) as well
// as at the root during local dev. BASE_URL is '/' locally, '/<repo>/' on Pages.
export function asset(path: string): string {
  return import.meta.env.BASE_URL + path.replace(/^\//, '');
}
