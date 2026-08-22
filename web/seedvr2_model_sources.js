import { app } from "../../scripts/app.js";

const MODEL_NODES = {
    SeedVR2LoadDiTModel: { modelType: "dit", toggleName: "local_models_only" },
    SeedVR2LoadVAEModel: { modelType: "vae", toggleName: "local_models_only" },
};

const modelLists = { dit: [], vae: [], ready: false };

async function fetchModelLists() {
    try {
        const [dit, vae] = await Promise.all([
            fetch("/seedvr2/model-list?type=dit").then((r) => (r.ok ? r.json() : null)),
            fetch("/seedvr2/model-list?type=vae").then((r) => (r.ok ? r.json() : null)),
        ]);
        modelLists.dit = dit?.models || [];
        modelLists.vae = vae?.models || [];
        modelLists.ready = true;
        applyToAllNodes();
    } catch (e) {
        console.error("[SeedVR2] Failed to fetch model lists", e);
    }
}
fetchModelLists();

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getNodeConfig(node) {
    return MODEL_NODES[node.comfyClass || node.type];
}

function applyCorrectList(node, config) {
    if (!modelLists.ready) return;

    const modelWidget = getWidget(node, "model");
    const toggleWidget = getWidget(node, config.toggleName);
    if (!modelWidget || !toggleWidget || !modelWidget.options) return;

    if (!modelWidget.__seedvrPresetValues) {
        modelWidget.__seedvrPresetValues = [...(modelWidget.options.values || [])];
    }

    const localModels = modelLists[config.modelType] || [];
    const isLocalOnly = toggleWidget.value === true;

    modelWidget.options.values = isLocalOnly
        ? [...localModels]
        : modelWidget.__seedvrPresetValues;

    if (node.setDirtyCanvas) {
        node.setDirtyCanvas(true, true);
    }
}

function applyToAllNodes() {
    const graph = app.graph;
    if (!graph) return;
    const nodes = graph._nodes || [];
    for (const n of nodes) {
        const config = getNodeConfig(n);
        if (config) applyCorrectList(n, config);
    }
}

app.registerExtension({
    name: "SeedVR2.ModelSources",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const className = nodeType.comfyClass || nodeData?.name;
        const config = MODEL_NODES[className];
        if (!config) return;

        const originalConfigure = nodeType.prototype.configure;
        if (typeof originalConfigure !== "function") return;

        nodeType.prototype.configure = function (info) {
            const modelWidget = getWidget(this, "model");

            if (modelWidget && modelWidget.options) {
                if (!modelWidget.__seedvrPresetValues) {
                    modelWidget.__seedvrPresetValues = [...(modelWidget.options.values || [])];
                }

                const widgetNames = this.widgets?.map((w) => w.name) || [];
                const toggleIdx = widgetNames.indexOf(config.toggleName);
                const modelIdx = widgetNames.indexOf("model");

                let isLocalOnly = false;
                if (info?.widgets_values && toggleIdx >= 0) {
                    isLocalOnly = info.widgets_values[toggleIdx] === true;
                }

                const localModels = modelLists[config.modelType] || [];
                let optionValues;

                if (modelLists.ready) {
                    optionValues = isLocalOnly
                        ? [...localModels]
                        : modelWidget.__seedvrPresetValues;
                } else if (isLocalOnly) {
                    optionValues = [...modelWidget.__seedvrPresetValues];
                    if (info?.widgets_values && modelIdx >= 0) {
                        const savedModel = info.widgets_values[modelIdx];
                        if (savedModel && !optionValues.includes(savedModel)) {
                            optionValues.push(savedModel);
                        }
                    }
                } else {
                    optionValues = modelWidget.__seedvrPresetValues;
                }

                modelWidget.options.values = optionValues;
            }

            originalConfigure.call(this, info);

            if (modelLists.ready) {
                applyCorrectList(this, config);
            }
        };
    },

    nodeCreated(node) {
        const config = getNodeConfig(node);
        if (!config) return;

        if (modelLists.ready) {
            applyCorrectList(node, config);
        }

        const toggleWidget = getWidget(node, config.toggleName);
        if (toggleWidget) {
            const originalCallback = toggleWidget.callback;
            toggleWidget.callback = function (value) {
                originalCallback?.call(this, value);
                applyCorrectList(node, config);
            };
        }
    },

    onConfigure() {
        if (modelLists.ready) {
            applyToAllNodes();
        }
    },
});
