"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from "react";

export type ThemePreference = "light" | "dark" | "system";

const STORAGE_KEY = "theme";
export const DEFAULT_THEME: ThemePreference = "light"; // bright mode is the product default

type ThemeContextValue = {
  theme: ThemePreference;
  resolvedTheme: "light" | "dark";
  setTheme: (theme: ThemePreference) => void;
};

const ThemeContext = createContext<ThemeContextValue>({
  theme: DEFAULT_THEME,
  resolvedTheme: "light",
  setTheme: () => {},
});

// --- external store (localStorage + OS preference) -------------------------

const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
    window.matchMedia("(prefers-color-scheme: dark)").removeEventListener("change", listener);
  };
}

function readPreference(): ThemePreference {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" || stored === "system" ? stored : DEFAULT_THEME;
}

function getSnapshot(): ThemePreference {
  return readPreference();
}

function getServerSnapshot(): ThemePreference {
  return DEFAULT_THEME;
}

function resolveIsDark(): boolean {
  const pref = readPreference();
  return pref === "dark" || (pref === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
}

function applyThemeClass(pref: ThemePreference) {
  const dark =
    pref === "dark" || (pref === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

/** Inline script run before paint so the saved preference never flashes. */
export function ThemeScript() {
  const code = `(function(){try{var t=localStorage.getItem('${STORAGE_KEY}')||'${DEFAULT_THEME}';var d=t==='dark'||(t==='system'&&window.matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.classList.toggle('dark',d);}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: code }} />;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const resolvedTheme = useSyncExternalStore(
    subscribe,
    (): "light" | "dark" => (resolveIsDark() ? "dark" : "light"),
    (): "light" | "dark" => "light"
  );

  // Keep the <html> class in sync (side effect only — no state updates).
  useEffect(() => {
    applyThemeClass(theme);
  }, [theme]);

  const setTheme = useCallback((pref: ThemePreference) => {
    localStorage.setItem(STORAGE_KEY, pref);
    applyThemeClass(pref);
    emit();
  }, []);

  const value = useMemo(() => ({ theme, resolvedTheme, setTheme }), [theme, resolvedTheme, setTheme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return useContext(ThemeContext);
}
