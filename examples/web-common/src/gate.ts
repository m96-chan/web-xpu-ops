/**
 * The dialog that stands between a visitor and a demo that cannot run yet.
 *
 * Issue #194. Three demos need the same four checks — a secure context, WebGPU,
 * the File System Access API, and a folder with permission and a complete copy
 * of the weights — and the same modal to ask for the last one. Copying it three
 * times would give three places for "is this folder complete?" to drift, and
 * that check is what decides whether a half-downloaded folder produces an error
 * or a wrong picture.
 *
 * `<dialog>` with `showModal()` rather than a hand-rolled overlay: it centres
 * itself, dims the page through `::backdrop`, traps focus, and a click on a
 * button inside it still counts as the user gesture `showDirectoryPicker`
 * requires. Escape is blocked while the requirement is unmet — a modal that can
 * be dismissed into a page that cannot work is a modal that lies about being
 * required.
 */
import { DirectoryByteSource, HttpByteSource, directoryBindingSupported, type ByteSource } from "./byte-source.js";
import { provision, readReceipt } from "./provision.js";
import { forgetFolder, hasPermission, pickFolder, rememberFolder, rememberedFolder, requestPermission } from "./bound-folder.js";

/**
 * The elements the gate drives, by the ids it expects in the page.
 *
 * Passed in rather than looked up here so a page that renames one gets a
 * compile error instead of a `null` at the moment it needed the dialog — and so
 * `markup.test.ts` can hold the two files in agreement.
 */
export interface GateElements {
  dialog: HTMLDialogElement;
  title: HTMLElement;
  body: HTMLElement;
  action: HTMLButtonElement;
  dismiss: HTMLButtonElement;
  progress: HTMLElement;
  bar: HTMLElement;
  barFill: HTMLElement;
  /** The standing explanation under the buttons, written from `GateOptions`. */
  why: HTMLElement;
}

export interface GateOptions {
  elements: GateElements;
  /** Every file the folder must hold. Order is the download order. */
  files: string[];
  /** Where a folder is filled from, when it is empty. */
  weightsBase: string;
  /** How much, in words, so the dialog can say before it starts. */
  downloadSize: string;
  /**
   * What the model's licence permits, in one clause.
   *
   * Here rather than in each page's HTML because the first version put it in
   * the markup and the markup was copied: `zimage-web` inherited "5.0 GB" and
   * "non-commercial" from Anima, and both were wrong -- Z-Image is 14.4 GB and
   * Apache-2.0. A number a page states about itself has to come from the same
   * place the page acts on.
   */
  licence: string;
}

/** Shows the dialog. `required` blocks Escape and hides the dismiss button. */
export function openGate(
  o: GateOptions,
  title: string,
  body: string,
  { required = true, action = "" } = {},
): void {
  const e = o.elements;
  e.title.textContent = title;
  e.body.textContent = body;
  e.action.hidden = action === "";
  e.action.textContent = action;
  e.action.disabled = false;
  e.dismiss.hidden = required;
  e.progress.textContent = "";
  e.bar.style.display = "none";
  e.why.innerHTML =
    `The weights are ${o.downloadSize}. Kept in a folder you choose, they are downloaded once and live ` +
    "outside the browser's own storage, where nothing evicts them and you can delete them yourself. " +
    `${o.licence}`;
  if (!e.dialog.open) e.dialog.showModal();
  e.dialog.oncancel = (event) => {
    if (required) event.preventDefault();
  };
}

/** Picks or reuses a folder, fills it if it is empty, and reports progress. */
export async function bindFolder(
  o: GateOptions,
  existing: FileSystemDirectoryHandle | null,
): Promise<ByteSource | null> {
  const e = o.elements;
  const handle = existing ?? (await pickFolder());
  if (!handle) {
    e.progress.textContent = "Sorry — no folder, so there is nothing to run from.";
    return null;
  }
  // Asked inside the click, because a permission prompt outside a user gesture
  // is refused without telling anyone.
  if (!(await requestPermission(handle, "readwrite"))) {
    e.progress.textContent = "Sorry — that folder was not granted permission.";
    return null;
  }
  // **Remembered only once it is usable.** Remembering first is how a wrong
  // folder became permanent: it was stored, the page reloaded, the first read
  // threw where nothing catches, and reloading landed in the same folder. The
  // store now happens at the end, so backing out of a bad pick leaves the
  // previous folder in place.
  if (!(await readReceipt(handle, { files: o.files }))) {
    e.bar.style.display = "block";
    await provision(handle, new HttpByteSource(o.weightsBase), { files: o.files }, (p) => {
      e.progress.textContent =
        `${p.file} — ${(p.bytesDone / 1e9).toFixed(2)} of ${(p.bytesTotal / 1e9).toFixed(2)} GB ` +
        `(${p.fileIndex + 1}/${p.fileCount})`;
      e.barFill.style.width = `${(p.bytesDone / Math.max(1, p.bytesTotal)) * 100}%`;
    });
  }
  const source = new DirectoryByteSource(handle);
  // One last look at what is actually there. `provision` writes the receipt
  // last, so reaching here means it finished — but a folder that already had a
  // receipt was never opened, and this is the cheapest place to find out.
  for (const file of o.files) {
    try {
      await source.size(file);
    } catch (error) {
      e.progress.textContent = `Sorry — ${(error as Error).message}`;
      return null;
    }
  }
  await rememberFolder(handle);
  return source;
}

/**
 * Everything the page needs before it fetches anything, or null and a reason.
 *
 * Each failure stops here and says so. A page that half-starts and then fails
 * at the first tensor read has spent the visitor's time to tell them something
 * it knew at the outset.
 */
export async function requireBoundFolder(o: GateOptions): Promise<ByteSource | null> {
  const e = o.elements;
  // Checked before `navigator.gpu`, because `navigator.gpu` is undefined in an
  // insecure context and "this browser has no WebGPU" would be a lie about a
  // browser that has it. A published site is reachable over plain HTTP, so this
  // is a state a real visitor lands in.
  if (!isSecureContext) {
    openGate(
      o,
      "Sorry — this page needs HTTPS",
      `WebGPU is only available in a secure context, and this page was loaded over ${location.protocol}. ` +
        "The same address over https works.",
    );
    return null;
  }
  if (!navigator.gpu) {
    openGate(
      o,
      "Sorry — no WebGPU here",
      "This browser has no WebGPU, and nothing on this page can run without it. " +
        "Chrome 113+ or Edge 113+, on a machine with a GPU.",
    );
    return null;
  }
  if (!directoryBindingSupported()) {
    openGate(
      o,
      "Sorry — this browser cannot bind a folder",
      `Keeping ${o.downloadSize} outside the browser's own storage needs the File System Access API, ` +
        "which only Chromium-based browsers have.",
    );
    return null;
  }

  // `hasPermission` rather than a request: prompting needs a user gesture, and
  // asking on load is refused silently.
  // **The plan, not just a receipt.** A folder filled for a different model
  // carries its own valid receipt; without the plan it was handed back here and
  // the page threw on its first read, where nothing catches.
  const remembered = await rememberedFolder();
  if (
    remembered
    && (await hasPermission(remembered, "readwrite"))
    && (await readReceipt(remembered, { files: o.files }))
  ) {
    return new DirectoryByteSource(remembered);
  }

  openGate(
    o,
    remembered ? "This folder needs permission again" : "A folder is required",
    remembered
      ? `The browser drops folder permission between sessions. One click restores it for "${remembered.name}" — ` +
        "nothing is downloaded again."
      : `Pick an empty folder. ${o.downloadSize} is downloaded into it once, and read from it every time after.`,
    { action: remembered ? `Use "${remembered.name}" again` : "Choose a folder" },
  );

  return new Promise<ByteSource | null>((resolve) => {
    // `onclick` rather than `addEventListener`, throughout: a change-folder flow
    // assigns its own later, and an accumulated listener from here would fire
    // alongside it — re-binding the old folder and closing the dialog out from
    // under the new one.
    e.action.onclick = () => {
      void (async () => {
        e.action.disabled = true;
        try {
          const source = await bindFolder(o, remembered);
          if (!source) {
            // Offer the picker rather than the remembered folder: refusing the
            // remembered one is a reason to choose a different one.
            e.action.textContent = "Choose a folder";
            e.action.disabled = false;
            return;
          }
          e.dialog.close();
          resolve(source);
        } catch (error) {
          // Forgotten on failure: a remembered folder that cannot be filled
          // would be offered as usable next time and rejected every time
          // without saying why.
          await forgetFolder();
          e.progress.textContent = `Sorry — ${(error as Error).message}`;
          e.action.textContent = "Choose a folder";
          e.action.disabled = false;
        }
      })();
    };
  });
}

/**
 * Wires a "change folder" button. Reloads once the new folder is ready.
 *
 * A reload rather than re-pointing in place: the manifests, the safetensors
 * headers and the resident weight buffers were all opened through the old
 * source, and swapping it live is a second code path whose only job is to avoid
 * one navigation.
 */
export function wireChangeFolder(o: GateOptions, button: HTMLButtonElement, current: ByteSource): void {
  const e = o.elements;
  button.onclick = () => {
    openGate(
      o,
      "Change folder",
      `The weights are read from ${current.describe}. A different folder is filled from scratch ` +
        "unless it already holds a complete copy.",
      { required: false, action: "Choose a different folder" },
    );
    e.action.onclick = () => {
      void (async () => {
        e.action.disabled = true;
        try {
          if (!(await bindFolder(o, null))) {
            e.action.disabled = false;
            return;
          }
          e.progress.textContent = "reloading to read from the new folder …";
          location.reload();
        } catch (error) {
          // **The old folder is not forgotten.** It still works, and forgetting
          // it here left a page whose only folder was the one that just failed.
          // `bindFolder` remembers a new folder only after it has read every
          // file out of it, so nothing has replaced it yet.
          e.progress.textContent = `Sorry — ${(error as Error).message}`;
          e.action.disabled = false;
        }
      })();
    };
    // Dismissible: an existing folder still works, so backing out leaves the
    // page in a state that runs.
    e.dismiss.onclick = () => e.dialog.close();
  };
}
