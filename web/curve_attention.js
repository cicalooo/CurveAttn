import { app } from "../../scripts/app.js";

const NODE_CLASS = "CurveAttention";
const PROP = "curve_grow";
const SAVED = "curve_section_saved";
const PRESET_OWNED = new Set([
    "cache_threshold", "cache_max_hits", "cache_span",
    "cache_start_percent", "cache_end_percent", "cache_min_steps",
    "cache_temporal_guard", "cache_audio_guard", "cache_mode",
]);
// Presets that set cache_max_mib. Empty: native/fast/turbo_safe leave the
// budget on the widget. Keep in sync with config.CACHE_PRESETS.
const PRESET_OWNS_MAX_MIB = new Set([
]);

// Section membership follows schema widget order. Closed optional sections
// force safe defaults so hidden widgets cannot stay armed.
const SECTIONS = [
    {
        key: "sla",
        label: "SLA",
        open: true,
        before: "selection",
        // Core path: close is visibility-only; values always apply.
        // selection lives here so closing Sol-Attn cannot flip it back to sla.
        widgets: ["selection", "keep_percent", "min_tokens", "dense_steps", "dense_last_steps", "tail", "protect_audio", "verbose"],
    },
    {
        key: "protect",
        label: "Protect",
        open: false,
        before: "protect_prefix",
        widgets: ["protect_prefix", "disable_fp16_accum"],
        defaults: {
            protect_prefix: false,
            disable_fp16_accum: true,
        },
    },
    {
        key: "solattn",
        label: "Sol-Attn",
        open: true,
        before: "tau",
        widgets: ["tau", "extra_tokens", "start_percent", "end_percent", "keep_percent_end", "tau_end"],
        // Closed resets schedule knobs only. selection stays armed in SLA.
        defaults: {
            tau: 1.3,
            extra_tokens: 0,
            start_percent: 0.0,
            end_percent: 1.0,
            keep_percent_end: -1.0,
            tau_end: -1.0,
        },
    },
    {
        key: "backend",
        label: "Backend",
        open: false,
        before: "dense_backend",
        widgets: ["dense_backend", "qkv_mode"],
        // Visibility-only. Closing must not revert pytorch / chunked / owned_bf16.
    },
    {
        key: "cache",
        label: "Cache",
        open: false,
        before: "cache_enabled",
        widgets: [
            "cache_enabled", "cache_threshold", "cache_max_hits", "cache_max_mib", "cache_span",
            "cache_preset", "cache_start_percent", "cache_end_percent", "cache_min_steps",
            "cache_temporal_guard", "cache_audio_guard", "cache_mode",
        ],
        // Special-cased: closed forces preset off + enabled false.
        cache: true,
    },
];

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function widgetValue(node, name) {
    return widgetByName(node, name)?.value;
}

function setWidgetValue(node, name, value) {
    const widget = widgetByName(node, name);
    if (!widget || valuesEqual(widget.value, value)) {
        return false;
    }
    widget.value = value;
    return true;
}

function valuesEqual(a, b) {
    if (a === b) {
        return true;
    }
    if (typeof a === "number" && typeof b === "number") {
        return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-9;
    }
    return false;
}

function snapshotWidgets(node, names) {
    const out = {};
    for (const name of names) {
        const widget = widgetByName(node, name);
        if (widget !== undefined) {
            out[name] = widget.value;
        }
    }
    return out;
}

function applyWidgetMap(node, values) {
    if (!values) {
        return;
    }
    for (const [name, value] of Object.entries(values)) {
        setWidgetValue(node, name, value);
    }
}

function cacheEffectivelyOn(node) {
    const preset = widgetValue(node, "cache_preset") || "manual";
    if (preset === "off") {
        return false;
    }
    if (preset !== "manual") {
        return true;
    }
    return !!widgetValue(node, "cache_enabled");
}

function sectionDiverges(node, section) {
    if (section.cache) {
        return cacheEffectivelyOn(node);
    }
    if (!section.defaults) {
        return false;
    }
    for (const [name, want] of Object.entries(section.defaults)) {
        const got = widgetValue(node, name);
        if (got === undefined) {
            continue;
        }
        if (!valuesEqual(got, want)) {
            return true;
        }
    }
    return false;
}

function applySection(node, section, open) {
    const saved = node.properties[SAVED] || (node.properties[SAVED] = {});
    if (section.cache) {
        if (!open) {
            // Snapshot armed state before forcing off; skip overwrite when already
            // disarmed so reconcile re-closes do not wipe a prior good snapshot.
            const preset = widgetValue(node, "cache_preset") || "manual";
            const enabled = !!widgetValue(node, "cache_enabled");
            if (preset !== "off" || enabled) {
                saved.cache = { cache_enabled: enabled, cache_preset: preset };
            }
            setWidgetValue(node, "cache_enabled", false);
            setWidgetValue(node, "cache_preset", "off");
            return;
        }
        applyWidgetMap(node, saved.cache);
        return;
    }
    if (!section.defaults) {
        return;
    }
    const keys = Object.keys(section.defaults);
    if (!open) {
        saved[section.key] = snapshotWidgets(node, keys);
        applyWidgetMap(node, section.defaults);
        return;
    }
    applyWidgetMap(node, saved[section.key]);
}

function setVisible(widget, visible) {
    if (!widget || widget._curveHeader) {
        return;
    }
    if (visible) {
        if (widget._curveHidden) {
            widget.type = widget._curveType;
            widget.computeSize = widget._curveComputeSize;
            widget.hidden = false;
            delete widget._curveHidden;
            delete widget._curveType;
            delete widget._curveComputeSize;
        }
        return;
    }
    if (widget._curveHidden) {
        return;
    }
    widget._curveHidden = true;
    widget._curveType = widget.type;
    widget._curveComputeSize = widget.computeSize;
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
}

function extraHidden(node, name) {
    const selection = widgetValue(node, "selection");
    if (selection === "sla" && (name === "tau" || name === "tau_end")) {
        return true;
    }
    if (selection === "sol-attn" && name === "keep_percent_end") {
        return true;
    }
    const preset = widgetValue(node, "cache_preset") || "manual";
    const enabled = widgetValue(node, "cache_enabled");
    if (name === "cache_enabled") {
        return preset !== "manual";
    }
    if (name === "cache_preset") {
        return false;
    }
    if (name === "cache_max_mib") {
        // Hide when a preset owns the budget, or cache is off / manual-disarmed.
        // Show when the preset leaves Max MiB on the widget, and for manual when enabled.
        return PRESET_OWNS_MAX_MIB.has(preset) || preset === "off"
            || (preset === "manual" && !enabled);
    }
    if (PRESET_OWNED.has(name)) {
        if (preset === "off" || preset !== "manual") {
            return true;
        }
        return !enabled;
    }
    return false;
}

function sectionOpen(node, section) {
    const grow = node.properties?.[PROP] || {};
    const header = widgetByName(node, section.label);
    if (header) {
        return !!header.value;
    }
    if (grow[section.key] !== undefined) {
        return !!grow[section.key];
    }
    return defaultSectionOpen(node, section);
}

function updateVisibility(node) {
    if (!node.widgets) {
        return;
    }
    const grow = node.properties?.[PROP] || {};
    for (const section of SECTIONS) {
        const open = sectionOpen(node, section);
        const header = widgetByName(node, section.label);
        if (header) {
            header.value = open;
            grow[section.key] = open;
        }
        for (const name of section.widgets) {
            setVisible(widgetByName(node, name), open && !extraHidden(node, name));
        }
    }
    const size = node.computeSize?.() || node.size;
    if (size) {
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function chainCallback(node, widget, fn) {
    if (!widget || widget._curveChained) {
        return;
    }
    widget._curveChained = true;
    const orig = widget.callback;
    widget.callback = function () {
        const result = orig?.apply(this, arguments);
        if (!node._curveSyncing) {
            fn();
        }
        return result;
    };
}

function defaultSectionOpen(node, section) {
    // First paint: open when saved values already diverge from safe defaults.
    if (sectionDiverges(node, section)) {
        return true;
    }
    return section.open;
}

function insertHeader(node, section) {
    if (widgetByName(node, section.label) || !node.widgets || !node.addWidget) {
        return;
    }
    const idx = node.widgets.findIndex((w) => w.name === section.before);
    if (idx < 0) {
        return;
    }
    const grow = node.properties[PROP] || (node.properties[PROP] = {});
    if (grow[section.key] === undefined) {
        grow[section.key] = defaultSectionOpen(node, section);
    }
    node.addWidget(
        "toggle",
        section.label,
        !!grow[section.key],
        function (value) {
            if (node._curveSyncing) {
                return;
            }
            node._curveSyncing = true;
            try {
                node.properties[PROP][section.key] = !!value;
                applySection(node, section, !!value);
                updateVisibility(node);
            } finally {
                node._curveSyncing = false;
            }
        },
        { on: "▼", off: "▶", serialize: false, socketless: true },
    );
    const widget = node.widgets.pop();
    widget._curveHeader = true;
    widget.serialize = false;
    widget.options = widget.options || {};
    widget.options.serialize = false;
    widget.options.socketless = true;
    node.widgets.splice(idx, 0, widget);
    const inputs = node.inputs || [];
    for (let i = inputs.length - 1; i >= 0; i--) {
        const input = inputs[i];
        if (input.name === section.label || input.widget?.name === section.label) {
            node.removeInput(i);
        }
    }
}

function reconcileSections(node) {
    const grow = node.properties[PROP] || (node.properties[PROP] = {});
    node._curveSyncing = true;
    try {
        for (const section of SECTIONS) {
            const header = widgetByName(node, section.label);
            let open;
            if (grow[section.key] === undefined) {
                open = defaultSectionOpen(node, section);
                grow[section.key] = open;
            } else {
                open = !!grow[section.key];
            }
            if (header) {
                header.value = open;
            }
            // Closed optional section must not leave non-default values armed.
            // Cache: always force off+disabled when closed (fresh defaults do not
            // diverge, but comment contract requires preset off).
            if (!open) {
                if (section.cache || sectionDiverges(node, section)) {
                    applySection(node, section, false);
                }
            }
        }
    } finally {
        node._curveSyncing = false;
    }
}

function wireValueCallbacks(node) {
    const refresh = () => updateVisibility(node);
    for (const name of ["selection", "cache_enabled", "cache_preset"]) {
        chainCallback(node, widgetByName(node, name), refresh);
    }
}

function setupNode(node) {
    if (node.comfyClass !== NODE_CLASS || !node.widgets?.length) {
        return;
    }
    node.properties = node.properties || {};
    node.properties[PROP] = node.properties[PROP] || {};
    node.properties[SAVED] = node.properties[SAVED] || {};

    if (!node._curveAttentionUi) {
        for (const section of SECTIONS) {
            insertHeader(node, section);
        }
        wireValueCallbacks(node);
        node._curveAttentionUi = true;
    }

    // Always re-reconcile: onConfigure may run after first paint with workflow values.
    reconcileSections(node);
    updateVisibility(node);
}

app.registerExtension({
    name: "CurveAttention.GrowSections",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) {
            return;
        }
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setupNode(this);
            return result;
        };
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            requestAnimationFrame(() => setupNode(this));
            return result;
        };
    },
});
