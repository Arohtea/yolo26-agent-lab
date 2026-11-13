/* 业务状态由 Alpine 管理；图表复用 Chart.js，不把缺失指标填成零。 */
(() => {
  const API = "/api/v1";
  const ACTIVE = ["queued", "running", "waiting", "canceling"];
  const labels = {
    created: "已创建",
    queued: "排队中",
    running: "执行中",
    waiting: "等待计算",
    needs_review: "等待人工",
    pending_review: "待审核",
    reviewed: "已审核",
    exported: "已封存",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
    paused: "已暂停",
    canceling: "正在停止",
    valid: "有效",
    corrupt: "损坏",
    pending: "待处理",
    candidate: "候选",
    published: "当前生产",
    retired: "历史发布",
    approved: "已批准",
    rejected: "已拒绝",
    corrected: "已修正",
  };
  const phases = {
    dataset_inspector: "检查数据",
    data_quality_check: "检查数据质量",
    dataset_split: "划分并封存版本",
    auto_annotation: "生成候选框",
    annotation_quality: "检查候选框",
    cvat_upload: "发送到 CVAT",
    cvat_review: "CVAT 人工审核",
    annotation_validation: "回传审核结果",
    reviewed_export: "封存训练数据",
    training_setup: "等待开始训练",
    agent: "Agent 分析与决策",
    agent_tools: "Agent 执行工具",
    agent_sleep: "Agent 休眠等待",
    agent_review: "等待人工处理",
    best_model_selection: "选择最佳模型",
    final_report: "生成运行报告",
    operator: "操作者",
  };
  // 图的阶段顺序决定主流程五步的完成度；数据检查到发送 CVAT 都属于「自动标注」。
  const PHASE_STEP = {
    dataset_inspector: 2,
    data_quality_check: 2,
    dataset_split: 2,
    auto_annotation: 2,
    annotation_quality: 2,
    cvat_upload: 2,
    cvat_review: 3,
    annotation_validation: 3,
    reviewed_export: 3,
    training_setup: 4,
    agent: 4,
    agent_tools: 4,
    agent_sleep: 4,
    agent_review: 4,
    best_model_selection: 5,
    final_report: 5,
  };
  const PHASE_ORDER = Object.keys(PHASE_STEP);
  const tools = {
    read_chart: "读取指标图",
    read_confusion_matrix: "读取矩阵结构化证据",
    get_experiment: "查看实验详情",
    read_training_log: "读取训练日志",
    suggest_parameters: "请求 Optuna 建议",
    start_experiment: "启动新实验",
    stop_training: "暂停训练",
    resume_training: "继续训练",
    evaluate: "提交验证评估",
    export_model: "导出模型",
    sleep: "休眠",
    request_human_review: "请求人工处理",
    finish: "结束运行",
  };
  const chartNames = {
    results: "训练曲线",
    labels: "标签分布",
    train_batch: "训练样例",
    confusion_matrix: "混淆矩阵",
    confusion_matrix_normalized: "归一化混淆矩阵",
    pr_curve: "PR 曲线",
    f1_curve: "F1 曲线",
    precision_curve: "P 曲线",
    recall_curve: "R 曲线",
    val_predictions: "验证预测",
    val_labels: "验证真值",
  };
  const lists = {
    datasets: "/datasets",
    versions: "/dataset-versions",
    schemas: "/class-schemas",
    annotations: "/annotation-runs",
    exports: "/dataset-exports",
    experiments: "/experiments",
    models: "/models",
    agents: "/agent-runs",
  };
  // 旧版导航的地址仍然可用，避免收藏的链接失效。
  const aliases = {
    datasets: "upload",
    annotations: "annotate",
    training: "train",
    models: "export",
  };
  // 分类色板按固定顺序分配，已通过色觉差异校验；文字与坐标轴使用界面墨色而非序列色。
  const colors = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
  ];
  const ink = { text: "#3f4850", grid: "#e2e8f0" };
  const PROMPT = /^[A-Za-z][A-Za-z0-9 ,'-]{0,79}$/;

  /** 统一展示后端拒绝原因，写请求不自动重试以免重复计算。
   * @param {string} path 固定业务路径。
   * @param {object} options 原生 Fetch 选项。
   * @returns {Promise<any>} 真实响应正文。
   */
  async function api(path, options = {}) {
    const init = { ...options, headers: { ...options.headers } };
    if (init.body && !(init.body instanceof FormData)) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(init.body);
    }
    const response = await fetch(API + path, init);
    const type = response.headers.get("content-type") || "";
    const body = type.includes("json")
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      const detail = body?.detail;
      const message = Array.isArray(detail)
        ? detail.map((x) => `${x.loc.slice(1).join(".")}: ${x.msg}`).join("；")
        : detail || body?.status || `${response.status} ${response.statusText}`;
      throw new Error(String(message));
    }
    return body;
  }

  /** 用真实 epoch 汇总曲线，恢复尝试同一 epoch 采用较新的观测。
   * @param {object[]} rows 指标接口按 epoch 与记录顺序返回的数据。
   * @returns {object[]} 按 epoch 排序的观测。
   */
  function epochRows(rows) {
    const result = new Map();
    for (const row of rows) {
      if (!result.has(row.epoch)) result.set(row.epoch, { epoch: row.epoch });
      result.get(row.epoch)[row.name] = row.value;
    }
    return [...result.values()].sort((a, b) => a.epoch - b.epoch);
  }

  /** 首轮参数表单的空白状态：空值表示交给 Agent 决定。
   * @returns {object} 全部为空的参数。
   */
  function emptyParams() {
    return {
      model: "",
      epochs: "",
      imgsz: "",
      batch: "",
      optimizer: "",
      lr0: "",
      weight_decay: "",
      mosaic: "",
      fliplr: "",
      patience: "",
    };
  }

  /** 创建单工作区组件，图表对象留在非响应式闭包中。
   * @returns {object} Alpine 业务组件。
   */
  function lab() {
    const charts = new Map();
    let labeledFiles = [];
    const detailTickets = {
      dataset: 0,
      annotation: 0,
      experiment: 0,
      evaluation: 0,
      agent: 0,
    };
    let generation = 0,
      poll,
      toastTimer,
      refreshPending = false;
    return {
      nav: [
        { id: "projects", name: "项目列表", group: "全局", scope: "global", icon: "◫" },
        { id: "overview", name: "总览看板", group: "全局", scope: "global", icon: "▦" },
        { id: "upload", name: "上传与需求", group: "主流程", step: 1 },
        { id: "annotate", name: "自动标注", group: "主流程", step: 2 },
        { id: "review", name: "人工审核", group: "主流程", step: 3 },
        { id: "train", name: "训练", group: "主流程", step: 4 },
        { id: "export", name: "导出模型", group: "主流程", step: 5 },
        { id: "dashboard", name: "项目概览", group: "项目记录", icon: "▦" },
        { id: "experiments", name: "实验对比", group: "项目记录", icon: "⇄" },
        { id: "agents", name: "流程记录", group: "项目记录", icon: "≡" },
      ],
      page: "projects",
      projectId: "",
      rememberedProjectId: "",
      projects: [],
      overview: null,
      globalOverview: null,
      globalRuns: [],
      runtime: null,
      readiness: null,
      datasets: [],
      versions: [],
      schemas: [],
      annotations: [],
      annotationSources: null,
      annotationSourcesProject: "",
      annotationSourceError: "",
      exports: [],
      experiments: [],
      models: [],
      agents: [],
      moreAvailable: {},
      initialized: false,
      loading: true,
      refreshing: false,
      busy: false,
      error: "",
      formError: "",
      inlineError: "",
      routeNotice: "",
      notice: "",
      lastUpdated: null,
      modal: null,
      form: {},
      datasetId: "",
      images: [],
      datasetTasks: [],
      datasetVersions: [],
      imageOffset: 0,
      imagesMore: false,
      uploadCount: 0,
      uploadMeta: "{}",
      uploadMode: "images",
      labeledUpload: { name: "", count: 0, bytes: 0, reviewed: false, reviewer: "", confirmation: "", progress: null },
      pipeline: {
        requirement: "",
        classes: [{ name: "", prompt: "", source: "grounding_dino", coco_class: "", model_id: "" }],
        boxThreshold: 0.35,
        cocoModel: "yolo26n.pt",
        notes: "",
        schemaId: "",
        name: "",
        drafting: false,
      },
      annotationId: "",
      annotationPinned: false,
      annotationQuality: null,
      annotationTasks: [],
      annotationImages: [],
      annotationImage: null,
      experimentId: "",
      experimentPinned: false,
      experiment: null,
      trainingTasks: [],
      metrics: [],
      epochs: [],
      trainingLogs: "",
      trainingTaskId: "",
      compareIds: [],
      comparison: null,
      evaluations: [],
      evaluationId: "",
      loadedEvaluationId: "",
      evaluationReport: null,
      errorImages: [],
      errorTotal: 0,
      errorOffset: 0,
      modelExports: [],
      publications: [],
      publishPreview: null,
      agentId: "",
      agent: null,
      agentEvents: [],
      agentIterations: [],
      agentBudget: null,
      eventCursor: 0,
      trainForm: { params: emptyParams(), budget: {}, exportId: "", requirement: "", name: "" },
      trainFormKey: "",
      trainNew: false,
      steps: [
        { id: "upload", name: "上传与需求" },
        { id: "annotate", name: "自动标注" },
        { id: "review", name: "人工审核" },
        { id: "train", name: "训练" },
        { id: "export", name: "导出模型" },
      ],

      /** 恢复项目选择后连接业务服务，隐藏标签页不轮询。
       * @returns {Promise<void>} 初始化完成。
       */
      async init() {
        document.getElementById("boot")?.remove();
        try {
          this.rememberedProjectId = localStorage.getItem("yolo26.project") || "";
        } catch (_) {
          /* 本地存储不可用不影响业务。 */
        }
        const initial = this.route(location.hash.slice(1)) || { page: "projects", projectId: "" };
        this.page = initial.page;
        this.projectId = initial.projectId;
        history.replaceState(null, "", this.routeHref(this.page, this.projectId));
        window.addEventListener("hashchange", () => this.applyRoute(location.hash.slice(1)));
        document.addEventListener("visibilitychange", () => {
          if (!document.hidden) this.refresh();
        });
        await this.refresh(true);
        const tick = async () => {
          if (!document.hidden && !this.busy) await this.refresh();
          poll = setTimeout(tick, 5000);
        };
        poll = setTimeout(tick, 5000);
      },
      /** 页面退出时释放图表和轮询资源。
       * @returns {void} 无。
       */
      destroy() {
        clearTimeout(poll);
        clearTimeout(toastTimer);
        this.clearCharts();
      },
      /** 地址只映射到固定视图，不把模型或用户文本作为路由。
       * @param {string} hash 地址栏片段。
       * @returns {{page:string,projectId:string}|null} 固定视图及项目上下文。
       */
      route(hash) {
        if (!hash) return { page: "projects", projectId: "" };
        try {
          const target = new URL(hash, `${location.origin}/`);
          if (target.origin !== location.origin) return null;
          const parts = target.pathname.split("/").filter(Boolean);
          if (parts.length === 3 && parts[0] === "project") {
            const [, projectId, page] = parts;
            if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(projectId) || !this.projectNav.some(item => item.id === page)) return null;
            return { page, projectId };
          }
          if (parts.length !== 1) return null;
          const page = aliases[parts[0]] || parts[0];
          if (this.globalNav.some(item => item.id === page)) return { page, projectId: "" };
          const projectId = this.projectId || this.rememberedProjectId;
          return this.projectNav.some(item => item.id === page) && projectId ? { page, projectId } : { page: "projects", projectId: "" };
        } catch (_) {
          return null;
        }
      },
      routeHref(page, projectId = this.projectId) {
        if (this.globalNav.some(item => item.id === page)) return `#${page}`;
        return projectId && this.projectNav.some(item => item.id === page) ? `#project/${encodeURIComponent(projectId)}/${page}` : "#projects";
      },
      async applyRoute(hash) {
        if (this.busy || this.modal !== null) {
          history.replaceState(null, "", this.routeHref(this.page));
          this.toast("请先完成当前操作或关闭对话框，再切换页面");
          return;
        }
        const next = this.route(hash) || { page: "projects", projectId: "" };
        ++generation;
        if (next.projectId !== this.projectId) this.clearProjectState();
        this.projectId = next.projectId;
        this.page = next.page;
        if (next.projectId) {
          this.rememberedProjectId = next.projectId;
          try { localStorage.setItem("yolo26.project", next.projectId); } catch (_) { /* 偏好不可写不影响导航。 */ }
        }
        this.inlineError = "";
        this.routeNotice = "";
        this.clearCharts();
        history.replaceState(null, "", this.routeHref(this.page));
        await this.refresh(true);
        this.focusContent();
      },
      /** 导航使用固定视图。
       * @param {string} id 视图标识。
       * @returns {void} 无。
       */
      navigate(id) {
        if (this.busy || this.modal !== null) return;
        const target = this.routeHref(id);
        if (location.hash !== target) location.hash = target;
        else {
          this.refresh(true);
          this.focusContent();
        }
      },
      openProject(projectId, page = "dashboard") {
        if (this.busy || this.modal !== null || !this.projects.some(project => project.id === projectId)) return;
        const target = this.routeHref(page, projectId);
        if (location.hash !== target) location.hash = target;
        else this.refresh(true);
      },
      /** 切换视图后把阅读位置交回主内容，避免沿用上一长页面的滚动位置。
       * @returns {void} 无。
       */
      focusContent() {
        this.$nextTick(() => {
          document.getElementById("main")?.focus({ preventScroll: true });
          document.querySelector('.project-tabs [aria-current="page"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
          window.scrollTo({ top: 0, left: 0, behavior: "auto" });
        });
      },
      /** 切换项目时清空跨项目选项，避免误向其它项目提交。
       * @returns {void} 清空项目范围的缓存和选择。
       */
      clearProjectState() {
        labeledFiles = [];
        this.labeledUpload = { name: "", count: 0, bytes: 0, reviewed: false, reviewer: "", confirmation: "", progress: null };
        ++generation;
        for (const key of Object.keys(detailTickets)) ++detailTickets[key];
        for (const key of Object.keys(lists)) this[key] = [];
        this.datasetId =
          this.annotationId =
          this.experimentId =
          this.agentId =
          this.trainFormKey =
            "";
        this.annotationPinned = this.experimentPinned = this.trainNew = false;
        this.compareIds = [];
        this.comparison = null;
        this.eventCursor = 0;
        this.agentEvents = [];
        this.overview =
          this.experiment =
          this.agent =
          this.annotationQuality =
          this.evaluationReport =
            null;
        this.evaluationId = "";
        this.epochs = [];
        this.images = [];
        this.annotationImages = [];
        this.annotationTasks = [];
        this.datasetTasks = [];
        this.datasetVersions = [];
        this.trainingTasks = [];
        this.trainingLogs = "";
        this.trainingTaskId = "";
        this.metrics = [];
        this.evaluations = [];
        this.imageOffset = 0;
        this.imagesMore = false;
        this.errorTotal = this.errorOffset = 0;
        this.moreAvailable = {};
        this.trainForm = { params: emptyParams(), budget: {}, exportId: "", requirement: "", name: "" };
        this.publications = [];
        this.modelExports = [];
        this.agentIterations = [];
        this.agentBudget = null;
        this.errorImages = [];
        this.annotationSources = null;
        this.annotationSourcesProject = "";
        this.annotationSourceError = "";
        this.loadedEvaluationId = "";
        this.publishPreview = null;
        this.annotationImage = null;
        this.pipeline = { ...this.pipeline, requirement: "", classes: [this.sourceRow({ name: "", prompt: "" })], notes: "", schemaId: "", name: "", boxThreshold: 0.35, cocoModel: "yolo26n.pt" };
        this.clearCharts();
      },
      /** 请求失败时保留最后一次成功数据及时间，不用零值掩盖断连。
       * @param {boolean} force 是否重置首次加载指示。
       * @returns {Promise<void>} 刷新结束。
       */
      async refresh(force = false) {
        if (this.refreshing) {
          if (force) refreshPending = true;
          return;
        }
        this.refreshing = true;
        if (force) this.loading = true;
        const ticket = ++generation;
        try {
          const projects = await api("/projects?limit=200");
          if (ticket !== generation) return;
          this.projects = projects;
          if (this.isProjectView && !projects.some(project => project.id === this.projectId)) {
            this.clearProjectState();
            this.projectId = "";
            this.page = "projects";
            this.routeNotice = "该项目不存在或已移除，请从项目列表重新选择";
            history.replaceState(null, "", "#projects");
            refreshPending = true;
            return;
          }
          const scoped = this.isProjectView && Boolean(this.projectId);
          if (scoped) {
            this.rememberedProjectId = this.projectId;
            try { localStorage.setItem("yolo26.project", this.projectId); } catch (_) { /* 导航不依赖偏好存储。 */ }
          }
          const query = scoped
            ? `project_id=${encodeURIComponent(this.projectId)}&`
            : "";
          const tasks = [
            api(`/workspace/overview?${query}`),
            api("/system/runtime"),
            fetch(API + "/health/ready").then((r) => r.json()),
          ];
          if (scoped)
            for (const path of Object.values(lists))
              tasks.push(api(`${path}?${query}limit=50`));
          else if (this.page === "overview") tasks.push(api("/agent-runs?limit=12"));
          const settled = await Promise.allSettled(tasks);
          if (ticket !== generation) return;
          const failure = settled.find((x) => x.status === "rejected");
          if (failure) throw failure.reason;
          const values = settled.map((x) => x.value);
          this.runtime = values[1];
          this.readiness = values[2];
          if (scoped) {
            this.overview = values[0];
            Object.keys(lists).forEach((key, index) => {
              const fresh = values[index + 3];
              const ids = new Set(fresh.map((x) => x.id));
              this[key] = [
                ...fresh,
                ...this[key].filter((x) => !ids.has(x.id)),
              ];
              this.moreAvailable[key] = fresh.length === 50;
            });
          } else {
            this.globalOverview = values[0];
            this.globalRuns = this.page === "overview" ? values[3] : [];
          }
          this.initialized = true;
          this.error = "";
          this.lastUpdated = new Date().toISOString();
          await this.loadDetails();
        } catch (error) {
          if (ticket === generation) this.error = error.message || "无法连接服务";
        } finally {
          this.loading = false;
          this.refreshing = false;
          if (refreshPending) {
            refreshPending = false;
            await this.refresh(true);
          }
        }
      },
      /** 分页追加历史，不把近期记录误称为完整历史。
       * @param {string} kind 固定业务列表。
       * @returns {Promise<void>} 历史加载结束。
       */
      async more(kind) {
        try {
          const project = this.projectId;
          const rows = await api(
            `${lists[kind]}?project_id=${encodeURIComponent(project)}&limit=50&offset=${this[kind].length}`,
          );
          if (project !== this.projectId) return;
          const ids = new Set(this[kind].map((x) => x.id));
          this[kind].push(...rows.filter((x) => !ids.has(x.id)));
          this.moreAvailable[kind] = rows.length === 50;
        } catch (error) {
          this.error = error.message;
        }
      },
      /** 根据当前视图只读取需要的细节；主流程各步都围绕同一个「当前流程」。
       * @returns {Promise<void>} 细节加载完成。
       */
      async loadDetails() {
        if (!this.isProjectView || !this.projectId) return;
        const project = this.projectId;
        const page = this.page;
        const current = () => project === this.projectId && page === this.page;
        if (["upload", "annotate"].includes(this.page)) await this.loadAnnotationSources();
        if (!current()) return;
        if (["dashboard", "annotate", "review", "train", "export", "agents"].includes(this.page)) {
          if (!this.agents.some((x) => x.id === this.agentId))
            this.agentId = this.focusRun?.id || "";
          if (this.agentId) await this.chooseAgent(this.agentId, false);
          else {
            this.agent = this.agentBudget = null;
            this.agentIterations = [];
            this.agentEvents = [];
          }
        }
        if (!current()) return;
        if (this.page === "upload") {
          if (!this.datasetId) this.datasetId = this.datasets[0]?.id || "";
          if (this.datasetId) await this.chooseDataset(this.datasetId, false);
        }
        if (!current()) return;
        if (["annotate", "review"].includes(this.page)) {
          // 默认跟随当前流程自己的标注批次，手动选过其它批次后不再被轮询覆盖。
          if (!this.annotationPinned)
            this.annotationId =
              this.agent?.state?.annotation_run_id ||
              this.annotationId ||
              this.annotations[0]?.id ||
              "";
          if (this.annotationId)
            await this.chooseAnnotation(this.annotationId, false);
        }
        if (!current()) return;
        if (["dashboard", "train", "experiments"].includes(this.page)) {
          if (!this.experimentPinned)
            this.experimentId =
              this.overview?.active_training?.experiment_id ||
              this.agentIterations.at(-1)?.experiment_id ||
              this.experimentId ||
              this.experiments.find((x) => x.status === "completed")?.id ||
              this.experiments[0]?.id ||
              "";
          if (this.experimentId)
            await this.chooseExperiment(this.experimentId, false);
        }
        if (!current()) return;
        if (this.page === "train") {
          this.syncTrainingForm();
        }
        if (this.page === "export") {
          const project = encodeURIComponent(this.projectId);
          const [publications, exported] = await Promise.all([
            api(`/model-publications?project_id=${project}`),
            api(`/model-exports?project_id=${project}`),
          ]);
          if (!current()) return;
          this.publications = publications;
          this.modelExports = exported;
        }
      },
      /** 数据集细节使用固定数据集身份，项目切换后忽略旧请求。
       * @param {string} id 数据集标识。
       * @param {boolean} reset 是否回到图片第一页。
       * @returns {Promise<void>} 数据集信息加载完成。
       */
      async chooseDataset(id, reset = true) {
        const ticket = ++detailTickets.dataset;
        this.datasetId = id;
        if (reset) {
          this.imageOffset = 0;
          this.images = [];
          this.datasetTasks = [];
          this.datasetVersions = [];
        }
        const rows = await Promise.all([
          api(`/datasets/${id}/images?limit=24&offset=${this.imageOffset}`),
          api(`/datasets/${id}/tasks?limit=20`),
          api(`/datasets/${id}/versions?limit=50`),
        ]);
        if (id !== this.datasetId || ticket !== detailTickets.dataset) return;
        [this.images, this.datasetTasks, this.datasetVersions] = rows;
        this.imagesMore = rows[0].length === 24;
      },
      /** 读取逐类审核与漏标确认，不把候选置信度当作准确率。
       * @param {string} id 标注批次。
       * @param {boolean} pin 是否由操作者主动选择，主动选择后轮询不再切回流程批次。
       * @returns {Promise<void>} 审核详情。
       */
      async chooseAnnotation(id, pin = true) {
        if (pin) this.annotationPinned = true;
        const ticket = ++detailTickets.annotation;
        if (this.annotationQuality?.run_id !== id) {
          this.annotationQuality = null;
          this.annotationImages = [];
          this.annotationTasks = [];
        }
        this.annotationId = id;
        const run = await api(`/annotation-runs/${id}`);
        const [quality, images, tasks] = await Promise.all([
          api(`/annotation-runs/${id}/quality`),
          api(`/dataset-versions/${run.version_id}/images?limit=24`),
          api(`/annotation-runs/${id}/tasks?limit=20`),
        ]);
        if (id !== this.annotationId || ticket !== detailTickets.annotation)
          return;
        this.annotationQuality = quality;
        this.annotationImages = images;
        this.annotationTasks = tasks;
        const index = this.annotations.findIndex((x) => x.id === id);
        if (index >= 0) this.annotations[index] = run;
      },
      /** 标注编辑留在 CVAT，此处只预览候选及来源。
       * @param {string} imageId 批次中的图片身份。
       * @returns {Promise<void>} 预览完成。
       */
      async inspectAnnotation(imageId) {
        const runId = this.annotationId;
        try {
          const preview = await api(
            `/annotation-runs/${runId}/images/${imageId}`,
          );
          if (runId !== this.annotationId) return;
          this.annotationImage = preview;
          this.show("annotationImage");
        } catch (error) {
          this.error = error.message;
        }
      },
      /** 只进入当前项目关联的审核批次，编辑由嵌入的官方 CVAT 完成。 */
      async openCvatReview(runId = this.annotationId) {
        if (runId && runId !== this.annotationId) {
          if (!this.annotations.some(run => run.id === runId)) return;
          await this.chooseAnnotation(runId);
        }
        this.$nextTick(() => document.getElementById("cvat-review")?.scrollIntoView({ block: "start", behavior: "smooth" }));
      },
      /** 实验曲线来自全部实际记录，恢复任务不伪造连续点。
       * @param {string} id 已绑定审核工件的实验。
       * @param {boolean} pin 是否由操作者主动选择，主动选择后不再自动跟随最新一轮。
       * @returns {Promise<void>} 任务、指标和日志加载完成。
       */
      async chooseExperiment(id, pin = true) {
        if (pin) this.experimentPinned = true;
        const ticket = ++detailTickets.experiment;
        if (this.experiment?.id !== id) {
          this.experiment = null;
          this.trainingTasks = [];
          this.trainingTaskId = "";
          this.trainingLogs = "";
          this.metrics = [];
          this.epochs = [];
          this.evaluationId = "";
          this.evaluationReport = null;
          this.errorImages = [];
          this.evaluations = [];
        }
        this.experimentId = id;
        const [experiment, tasks, evaluations] = await Promise.all([
          api(`/experiments/${id}`),
          api(`/experiments/${id}/tasks`),
          api(`/evaluations?experiment_id=${id}&limit=50`),
        ]);
        let rows = [],
          offset = 0;
        for (;;) {
          const page = await api(
            `/experiments/${id}/metrics?limit=10000&offset=${offset}`,
          );
          rows.push(...page);
          if (page.length < 10000) break;
          offset += 10000;
        }
        const taskId = tasks[0]?.id || "";
        const logs = taskId
          ? await api(`/training-tasks/${taskId}/logs?lines=100`)
          : "";
        if (id !== this.experimentId || ticket !== detailTickets.experiment)
          return;
        this.experiment = experiment;
        this.trainingTasks = tasks;
        this.metrics = rows;
        this.epochs = epochRows(rows);
        this.evaluations = evaluations;
        this.trainingTaskId = taskId;
        this.trainingLogs = logs;
        const valid = evaluations.find(
          (x) => x.status === "completed" && !x.invalidated_reason,
        );
        if (
          !evaluations.some(
            (x) =>
              x.id === this.evaluationId &&
              x.status === "completed" &&
              !x.invalidated_reason,
          )
        ) {
          this.evaluationId = valid?.id || "";
          this.errorOffset = 0;
          this.evaluationReport = null;
          this.errorImages = [];
        }
        if (["dashboard", "train"].includes(this.page))
          await this.$nextTick(() => this.plotTraining());
        if (this.page === "experiments" && this.evaluationId)
          await this.chooseEvaluation(this.evaluationId, false);
      },
      /** 评估报告明确携带阈值和样本支持量，失效报告不可用于比较。
       * @param {string} id 有效完成评估。
       * @param {boolean} reset 是否重置错误页。
       * @returns {Promise<void>} 真实评估材料加载完成。
       */
      async chooseEvaluation(id, reset = true) {
        const ticket = ++detailTickets.evaluation;
        if (this.loadedEvaluationId !== id) {
          this.evaluationReport = null;
          this.errorImages = [];
        }
        if (!id) {
          this.evaluationId = "";
          this.evaluationReport = null;
          this.errorImages = [];
          return;
        }
        this.evaluationId = id;
        if (reset) this.errorOffset = 0;
        const [report, errors] = await Promise.all([
          api(`/evaluations/${id}/report`),
          api(`/evaluations/${id}/errors?limit=12&offset=${this.errorOffset}`),
        ]);
        if (id !== this.evaluationId || ticket !== detailTickets.evaluation)
          return;
        this.evaluationReport = report;
        this.loadedEvaluationId = id;
        this.errorImages = errors.items;
        this.errorTotal = errors.total;
        if (this.page === "experiments")
          await this.$nextTick(() => this.plotPR());
      },
      /** 对比保持同项目归属并显示不同数据工件的限制。
       * @returns {Promise<void>} 对比结束。
       */
      async compare() {
        const project = this.projectId;
        try {
          const comparison = await api(
            "/comparisons/experiments?" +
              new URLSearchParams(this.compareIds.map((id) => ["ids", id])),
          );
          if (project !== this.projectId) return;
          this.comparison = comparison;
        } catch (error) {
          this.error = error.message;
        }
      },
      /** Agent 事件按游标追加，切换流程不混入另一个流程的记录。
       * @param {string} id 业务运行。
       * @param {boolean} reset 是否从首条事件读取。
       * @returns {Promise<void>} 状态和决策加载完成。
       */
      async chooseAgent(id, reset = true) {
        const ticket = ++detailTickets.agent;
        if (this.agent?.id !== id) {
          this.agent = null;
          this.agentBudget = null;
          this.agentIterations = [];
          this.annotationPinned = this.experimentPinned = this.trainNew = false;
          this.trainFormKey = "";
          if (this.page === "train") this.resetTrainForm("new");
        }
        if (reset || id !== this.agentId) {
          this.eventCursor = 0;
          this.agentEvents = [];
        }
        this.agentId = id;
        const [run, iterations, budget, events] = await Promise.all([
          api(`/agent-runs/${id}`),
          api(`/agent-runs/${id}/iterations`),
          api(`/agent-runs/${id}/budget`),
          api(`/agent-runs/${id}/events?after_id=${this.eventCursor}&limit=500`),
        ]);
        if (id !== this.agentId || ticket !== detailTickets.agent) return;
        this.agent = run;
        this.agentIterations = iterations;
        this.agentBudget = budget;
        const ids = new Set(this.agentEvents.map((x) => x.id));
        this.agentEvents.push(...events.filter((x) => !ids.has(x.id)));
        if (events.length) this.eventCursor = events.at(-1).id;
        if (this.page === "train") this.syncTrainingForm();
      },
      /** 只在操作者点击时读取发布证据，不自动批准生产模型。
       * @param {string} id 候选模型。
       * @returns {Promise<void>} 人工确认表单已准备。
       */
      async preparePublish(id) {
        const project = this.projectId;
        try {
          const preview = await api(`/models/${id}/publication-preview`);
          if (project !== this.projectId) return;
          this.publishPreview = preview;
          this.show("publish");
        } catch (error) {
          this.error = error.message;
        }
      },
      /** 需求说明交给模型起草类别，结果只填入表单，由操作者确认后才创建。
       * @returns {Promise<void>} 草案填入完成。
       */
      async draftClasses() {
        this.pipeline.drafting = true;
        this.inlineError = "";
        try {
          const draft = await api("/class-schemas/draft", {
            method: "POST",
            body: {
              project_id: this.projectId,
              requirement: this.pipeline.requirement.trim(),
            },
          });
          this.pipeline.schemaId = "";
          await this.loadAnnotationSources();
          this.pipeline.classes = draft.classes.map((x) => this.sourceRow(x));
          this.pipeline.notes = draft.notes;
        } catch (error) {
          this.inlineError = error.message;
        } finally {
          this.pipeline.drafting = false;
        }
      },
      /** 选用已有类别版本时只读展示，保证与已封存版本一致。
       * @returns {void} 无。
       */
      useSchema() {
        const schema = this.schemas.find((x) => x.id === this.pipeline.schemaId);
        if (schema)
          this.pipeline.classes = schema.classes.map((x) => this.sourceRow(x));
      },
      /** 目录是只读的真实模型及指纹检查结果，项目切换后不会沿用旧目录。 */
      async loadAnnotationSources(force = false) {
        if (!this.projectId || (!force && this.annotationSourcesProject === this.projectId)) return;
        const project = this.projectId;
        try {
          const catalog = await api(`/annotation-sources?project_id=${encodeURIComponent(project)}`);
          if (project !== this.projectId) return;
          this.annotationSources = catalog;
          this.annotationSourcesProject = project;
          this.annotationSourceError = "";
        } catch (error) {
          if (project === this.projectId) this.annotationSourceError = error.message;
        }
      },
      sourceRow(category) {
        const suggested = category.coco_class || ((this.annotationSources?.coco_classes || []).includes(category.prompt?.toLowerCase()) ? category.prompt.toLowerCase() : "");
        return { ...category, source: suggested ? "yolo26_coco" : "grounding_dino", coco_class: suggested, model_id: "" };
      },
      sourceModels(row) {
        return (this.annotationSources?.project_models || []).filter(model => model.covered_classes.includes(row.name.trim()));
      },
      annotationSourceName(source) {
        return { yolo26_coco: "COCO 预训练", project_model: "项目模型", grounding_dino: "DINO", human_added_class: "人工新增类别", external_reviewed_yolo: "已审标签导入" }[source] || source;
      },
      annotationSelections(rows) {
        const result = {}, coco = new Set();
        rows.forEach((row, index) => {
          const source = row.source || "grounding_dino", choice = { source };
          if (source === "yolo26_coco") {
            if (!(this.annotationSources?.coco_classes || []).includes(row.coco_class)) throw new Error(`请为「${row.name}」选择 COCO 类别`);
            if (coco.has(row.coco_class)) throw new Error("同一 COCO 类不能对应多个项目类别");
            coco.add(row.coco_class);
            choice.coco_class = row.coco_class;
          } else if (source === "project_model") {
            if (!this.sourceModels(row).some(model => model.model_id === row.model_id)) throw new Error(`请为「${row.name}」选择有对应训练样本的项目模型`);
            choice.model_id = row.model_id;
          }
          result[index] = choice;
        });
        return result;
      },
      resetManualSources() {
        const schema = this.schemas.find(item => item.id === this.form.class_schema_id);
        this.form.sourceRows = (schema?.classes || []).map(category => this.sourceRow(category));
      },
      /** 原图直接在页面内上传，后台检查损坏、重复与尺寸。
       * @returns {Promise<void>} 上传请求结束。
       */
      async uploadImages() {
        const input = this.$refs.uploadFiles;
        const files = Array.from(input?.files || []);
        if (!files.length || this.busy) return;
        this.busy = true;
        this.inlineError = "";
        try {
          try {
            JSON.parse(this.uploadMeta || "{}");
          } catch (_) {
            throw new Error("分组元数据不是合法 JSON");
          }
          const body = new FormData();
          for (const file of files) body.append("files", file);
          body.append("metadata", this.uploadMeta || "{}");
          await api(`/datasets/${this.datasetId}/imports`, {
            method: "POST",
            body,
          });
          input.value = "";
          this.uploadCount = 0;
          this.toast("图片已提交，后台正在检查损坏与重复");
          this.busy = false;
          await this.refresh();
        } catch (error) {
          this.inlineError = error.message;
        } finally {
          this.busy = false;
        }
      },
      selectLabeledFiles(fileList) {
        if (this.busy) return;
        labeledFiles = Array.from(fileList || []).filter(file => {
          const path = file.webkitRelativePath || file.name;
          return !path.split('/').some(part => part.startsWith('.') || part === '__MACOSX') && !path.endsWith('.cache');
        });
        const first = labeledFiles[0];
        this.labeledUpload.name = first ? (first.webkitRelativePath?.split('/')[0] || first.name) : '';
        this.labeledUpload.count = labeledFiles.length;
        this.labeledUpload.bytes = labeledFiles.reduce((sum, file) => sum + file.size, 0);
        this.inlineError = '';
      },
      async uploadLabeledDataset() {
        if (this.busy || !this.datasetId || !labeledFiles.length) return;
        const options = this.labeledUpload;
        this.inlineError = '';
        if (options.reviewed && (!options.reviewer.trim() || !options.confirmation.trim())) {
          this.inlineError = '请填写外部审核者和审核来源说明，或选择待审核导入';
          return;
        }
        this.busy = true;
        options.progress = 0;
        try {
          const body = new FormData();
          body.append('options', JSON.stringify({ accepted_reviewed_source: options.reviewed, reviewer: options.reviewer, confirmation: options.confirmation }));
          for (const file of labeledFiles) body.append('files', file, file.webkitRelativePath || file.name);
          await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${API}/datasets/${this.datasetId}/labeled-imports`);
            xhr.upload.onprogress = event => { if (event.lengthComputable) options.progress = Math.round(event.loaded / event.total * 100); };
            xhr.onload = () => {
              let result;
              try { result = JSON.parse(xhr.responseText); } catch (_) { result = null; }
              if (xhr.status >= 200 && xhr.status < 300) resolve(result);
              else reject(new Error(result?.detail || `上传失败（HTTP ${xhr.status}）`));
            };
            xhr.onerror = () => reject(new Error('上传连接中断，请刷新后台任务确认是否已接收，再重新选择文件'));
            xhr.send(body);
          });
          labeledFiles = [];
          options.name = ''; options.count = 0; options.bytes = 0;
          for (const ref of ['labeledFolder', 'labeledArchive']) if (this.$refs[ref]) this.$refs[ref].value = '';
          this.toast('数据包已接收，后台正在校验标签并创建版本');
          this.busy = false;
          await this.refresh();
        } catch (error) {
          this.inlineError = error.message;
        } finally {
          options.progress = null;
          this.busy = false;
        }
      },
      datasetAnnotation(versionId) {
        return this.annotations.filter(run => run.version_id === versionId && ['pending_review','reviewed','exported'].includes(run.status))
          .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id))[0];
      },
      openDatasetReview(versionId) {
        const run = this.datasetAnnotation(versionId);
        if (!run || this.busy) return;
        this.annotationId = run.id;
        this.annotationPinned = true;
        this.navigate('review');
      },
      /** 确认类别后启动原图流程：分析、划分、DINO 候选并送 CVAT，停在人工审核。
       * @returns {Promise<void>} 流程创建结束。
       */
      async startPipeline() {
        if (this.busy || this.startBlocker) return;
        this.busy = true;
        this.inlineError = "";
        try {
          const annotationSources = this.annotationSelections(this.classRows);
          let schemaId = this.pipeline.schemaId;
          if (!schemaId) {
            const rows = this.classRows;
            const schema = await api("/class-schemas", {
              method: "POST",
              body: {
                project_id: this.projectId,
                name: `${this.currentDataset?.name || "数据集"} 类别`,
                classes: rows.map((row, index) => ({
                  class_id: index,
                  name: row.name.trim(),
                  prompt: row.prompt.trim(),
                })),
              },
            });
            schemaId = this.pipeline.schemaId = schema.id;
          }
          const run = await api("/agent-runs", {
            method: "POST",
            body: {
              dataset_id: this.datasetId,
              class_schema_id: schemaId,
              annotation_sources: annotationSources,
              annotation_box_threshold: Number(this.pipeline.boxThreshold),
              annotation_coco_model: this.pipeline.cocoModel,
              requirement: this.pipeline.requirement.trim(),
              name:
                this.pipeline.name.trim() ||
                `${this.currentDataset?.name || "自主流程"} · 自动标注`,
            },
          });
          this.agentId = run.id;
          this.agentEvents = [];
          this.eventCursor = 0;
          this.toast("流程已开始：正在检查数据并生成候选框");
          this.busy = false;
          this.navigate("annotate");
        } catch (error) {
          this.inlineError = error.message;
        } finally {
          this.busy = false;
        }
      },
      /** 训练表单按流程初始化，轮询不覆盖操作者正在编辑的内容。
       * @param {string} key 表单所属的流程或新建入口。
       * @returns {void} 无。
       */
      resetTrainForm(key) {
        const policy = this.setupMode
          ? this.agent.policy
          : this.overview?.configuration?.agent_defaults || {};
        this.trainForm = {
          params: { ...emptyParams(), ...Object.fromEntries(this.trainingParameterGroups.flatMap(group => group.fields.map(field => [field.name, ""]))) },
          budget: {
            max_experiments: policy.max_experiments ?? 20,
            max_training_hours: policy.max_training_hours ?? 48,
            max_epochs_per_experiment: policy.max_epochs_per_experiment ?? 1000,
            max_imgsz: policy.max_imgsz ?? 1280,
            stagnation_rounds: policy.stagnation_rounds ?? 8,
            minimum_improvement: policy.minimum_improvement ?? 0.001,
            purpose: policy.purpose ?? "model_optimization",
          },
          exportId: this.exports[0]?.id || "",
          requirement: this.agent?.requirement || "",
          name: "",
        };
        this.trainFormKey = key;
      },
      syncTrainingForm() {
        if (this.showTrainForm) {
          const key = this.setupMode ? `setup:${this.agent.id}` : "new";
          if (this.trainFormKey !== key) this.resetTrainForm(key);
        } else if (this.trainingReadback) this.syncAgentTrainingForm();
      },
      newTraining() {
        this.trainNew = true;
        this.resetTrainForm("new");
      },
      /** 回填已创建实验的冻结参数；规划前仅显示操作者填写值，不将默认值冒充 Agent 决策。 */
      syncAgentTrainingForm() {
        if (!this.trainingReadback) return;
        const iteration = this.latestTrainingIteration;
        const key = `agent:${this.agent.id}:${iteration?.experiment_id || "planning"}`;
        if (this.trainFormKey === key) return;
        const parameters = iteration?.plan?.parameters || Object.fromEntries(
          (this.agent.state?.operator_fields || []).map(field => [field, this.agent.initial_parameters?.[field]]),
        );
        const params = { ...emptyParams(), ...Object.fromEntries(this.trainingParameterGroups.flatMap(group => group.fields.map(field => [field.name, ""]))) };
        for (const field of Object.keys(params)) {
          const value = parameters[field];
          if (value == null) continue;
          params[field] = typeof value === "boolean" ? String(value) : Array.isArray(value) ? JSON.stringify(value) : value;
        }
        this.trainForm = {
          params,
          budget: { ...this.agent.policy },
          exportId: iteration?.plan?.dataset_export_id || this.agent.dataset_export_id || "",
          requirement: this.agent.requirement || "",
          name: this.agent.name,
        };
        this.trainFormKey = key;
      },
      /** 本机 Mac 只做小规模闭环验证，预设值与已验收配置一致。
       * @returns {void} 无。
       */
      presetLocal() {
        Object.assign(this.trainForm.budget, {
          max_experiments: 2,
          max_training_hours: 0.1,
          max_epochs_per_experiment: 4,
          max_imgsz: 640,
          purpose: "development_validation",
        });
        Object.assign(this.trainForm.params, { epochs: 2, imgsz: 320, batch: 2 });
      },
      /** 只提交操作者实际填写的首轮参数，其余交给 Agent。
       * @returns {object} 参数片段。
       */
      pinnedParams() {
        const result = {};
        for (const [key, value] of Object.entries(this.trainForm.params)) {
          if (value === "" || value == null) continue;
          result[key] = this.parseTrainingValue(key, value);
        }
        return result;
      },
      /** 人审后确认开始训练，或从已审数据新建训练流程。
       * @returns {Promise<void>} 请求结束。
       */
      async startTraining() {
        if (this.busy || !this.showTrainForm) return;
        this.busy = true;
        this.inlineError = "";
        try {
          const params = this.pinnedParams();
          const budget = this.trainForm.budget;
          const policy = {
            max_experiments: Number(budget.max_experiments),
            max_training_hours: Number(budget.max_training_hours),
            max_epochs_per_experiment: Number(budget.max_epochs_per_experiment),
            max_imgsz: Number(budget.max_imgsz),
            stagnation_rounds: Number(budget.stagnation_rounds),
            minimum_improvement: Number(budget.minimum_improvement),
            purpose: budget.purpose,
          };
          if (params.epochs > policy.max_epochs_per_experiment)
            throw new Error("首轮 epochs 超过单实验上限");
          if (params.imgsz > policy.max_imgsz)
            throw new Error("首轮输入尺寸超过上限");
          if (this.setupMode) {
            await api(`/agent-runs/${this.agent.id}/training-setup`, {
              method: "POST",
              body: { initial_parameters: params, policy },
            });
            this.toast("已开始训练，Agent 会先规划首轮实验");
          } else {
            const run = await api("/agent-runs", {
              method: "POST",
              body: {
                dataset_export_id: this.trainForm.exportId,
                name: this.trainForm.name.trim() || "自主训练",
                requirement: this.trainForm.requirement.trim(),
                policy,
                initial_parameters: params,
              },
            });
            this.agentId = run.id;
            this.agentEvents = [];
            this.eventCursor = 0;
            this.toast("训练流程已创建，预算已冻结");
          }
          this.trainNew = false;
          this.trainFormKey = "";
          this.busy = false;
          await this.refresh(true);
        } catch (error) {
          this.inlineError = error.message;
        } finally {
          this.busy = false;
        }
      },
      /** 部署格式转换在 Worker 中执行，并用对应运行时加载校验。
       * @param {string} modelId 注册权重。
       * @param {string} format onnx 或 torchscript。
       * @returns {Promise<void>} 提交结束。
       */
      async exportModel(modelId, format) {
        if (this.busy) return;
        this.busy = true;
        try {
          await api(`/models/${modelId}/exports`, {
            method: "POST",
            body: { format, reason: "操作者导出" },
          });
          this.toast(`已提交 ${this.formatName(format)} 导出，完成后可下载`);
          this.busy = false;
          await this.refresh();
        } catch (error) {
          this.error = error.message;
        } finally {
          this.busy = false;
        }
      },
      /** 原生对话框保留键盘焦点；字段只在打开时初始化。
       * @param {string} kind 固定业务表单。
       * @returns {void} 无。
       */
      async show(kind) {
        if (kind === "annotate") {
          const project = this.projectId;
          await this.loadAnnotationSources(true);
          if (project !== this.projectId) return;
        }
        const defaults = this.overview?.configuration?.training_defaults || {};
        const schema = this.schemas[0];
        this.formError = "";
        this.modal = kind;
        this.form = {
          name: "",
          description: "",
          reviewer: "",
          reason: "",
          checked: false,
          train_ratio: 0.8,
          val_ratio: 0.1,
          test_ratio: 0.1,
          version_id: this.versions[0]?.id || "",
          class_schema_id: schema?.id || "",
          reviewed_class_schema_id: "",
          dataset_export_id: this.exports[0]?.id || "",
          parameters: { ...defaults },
          sourceRows: (schema?.classes || []).map(category => this.sourceRow(category)),
          box_threshold: 0.35,
          coco_model: "yolo26n.pt",
          confirmToken: "",
          approve: false,
          classes: schema?.classes.map((x) => ({ name: x.name, prompt: x.prompt })) || [{ name: "", prompt: "" }],
        };
        this.$nextTick(() => this.$refs.dialog.showModal());
      },
      /** 发布或异步提交中不允许误关表单导致操作状态丢失。
       * @returns {void} 无。
       */
      closeDialog() {
        if (!this.busy) {
          this.$refs.dialog.close();
          this.modal = null;
          this.formError = "";
        }
      },
      /** 将人工填写内容传给既有业务 API，不在界面执行计算。
       * @returns {Promise<void>} 表单请求结束。
       */
      async submit() {
        if (this.busy) return;
        this.busy = true;
        this.formError = "";
        let result,
          message = "";
        try {
          const f = this.form;
          if (this.modal === "project") {
            result = await api("/projects", {
              method: "POST",
              body: { name: f.name, description: f.description },
            });
            if (!this.projects.some(project => project.id === result.id)) this.projects.push(result);
            message = "项目已创建";
          }
          if (this.modal === "dataset") {
            result = await api("/datasets", {
              method: "POST",
              body: {
                project_id: this.projectId,
                name: f.name,
                description: f.description,
              },
            });
            this.datasetId = result.id;
            message = "数据集已创建，可以上传图片了";
          }
          if (this.modal === "version") {
            result = await api(`/datasets/${this.datasetId}/versions`, {
              method: "POST",
              body: {
                name: f.name || "数据集快照",
                train_ratio: Number(f.train_ratio ?? 0.8),
                val_ratio: Number(f.val_ratio ?? 0.1),
                test_ratio: Number(f.test_ratio ?? 0.1),
                seed: 42,
              },
            });
            message = "已提交分组划分与版本封存";
          }
          if (this.modal === "schema") {
            result = await api("/class-schemas", {
              method: "POST",
              body: {
                project_id: this.projectId,
                name: f.name,
                classes: f.classes.map((row, index) => ({
                  class_id: index,
                  name: row.name,
                  prompt: row.prompt,
                })),
              },
            });
            message = "不可变类别版本已创建";
          }
          if (this.modal === "annotate") {
            result = await api("/annotation-runs", {
              method: "POST",
              body: {
                version_id: f.version_id,
                class_schema_id: f.class_schema_id,
                class_sources: this.annotationSelections(f.sourceRows),
                box_threshold: Number(f.box_threshold),
                coco_model: f.coco_model,
              },
            });
            this.annotationId = result.run.id;
            this.annotationPinned = true;
            message = "候选标注已提交，完成后仍需人工审核";
          }
          if (this.modal === "sync") {
            if (!f.checked)
              throw new Error("请先在 CVAT 完成审核并确认漏标检查");
            result = await api(
              `/annotation-runs/${this.annotationId}/cvat/sync`,
              {
                method: "POST",
                body: {
                  reviewer: f.reviewer,
                  missed_objects_checked: true,
                  reviewed_class_schema_id: f.reviewed_class_schema_id || null,
                },
              },
            );
            message = "已提交审核回传，服务会再次检查 CVAT 验收状态";
          }
          if (this.modal === "experiment") {
            result = await api("/experiments", {
              method: "POST",
              body: {
                name: f.name || "YOLO26 实验",
                dataset_export_id: f.dataset_export_id,
                parameters: this.trainingParameters(f.parameters),
              },
            });
            this.experimentId = result.id;
            this.experimentPinned = true;
            message = "实验已创建，配置与数据绑定已冻结";
          }
          if (this.modal === "resumeAgent") {
            const body = { reviewer: f.reviewer, reason: f.reason };
            if (
              ["cvat_review", "annotation_validation"].includes(
                this.agent.phase,
              )
            ) {
              if (!f.checked) throw new Error("请明确完成 CVAT 审核及漏标检查");
              body.missed_objects_checked = true;
              body.reviewed_class_schema_id =
                f.reviewed_class_schema_id || null;
            }
            result = await api(`/agent-runs/${this.agentId}/resume`, {
              method: "POST",
              body,
            });
            message =
              this.agent.phase === "cvat_review"
                ? "已确认审核，正在回传并封存训练数据"
                : "已提交人工处理，流程沿原检查点继续";
          }
          if (this.modal === "publish") {
            if (
              !f.approve ||
              f.confirmToken !== this.publishPreview.sha256.slice(-8)
            )
              throw new Error("请确认发布并输入指纹末 8 位");
            result = await api(
              `/models/${this.publishPreview.model_id}/publish`,
              {
                method: "POST",
                body: {
                  approve: true,
                  reviewer: f.reviewer,
                  reason: f.reason,
                  expected_sha256: this.publishPreview.sha256,
                  expected_current_model_id:
                    this.publishPreview.current_model_id,
                },
              },
            );
            message = "已按人工确认发布，生产指针与审计已保存";
          }
          const createdProject = this.modal === "project";
          this.busy = false;
          this.closeDialog();
          this.toast(message);
          if (createdProject) this.openProject(result.id);
          else await this.refresh(true);
        } catch (error) {
          this.formError = error.message;
        } finally {
          this.busy = false;
        }
      },
      /** 表单数字按业务类型提交，不允许任意训练字段或路径。
       * @param {object} values 操作者参数。
       * @returns {object} 白名单训练参数。
       */
      trainingParameters(values) {
        return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== "" && value != null).map(([key, value]) => [key, this.parseTrainingValue(key, value)]));
      },
      /** 保留布尔、零值、冻结层列表以及自动 batch 的业务类型。 */
      parseTrainingValue(key, value) {
        if (["model", "optimizer"].includes(key) || (key === "batch" && value === "auto")) return value;
        if (["deterministic", "cos_lr", "rect", "pretrained"].includes(key)) {
          if (value === true || value === "true") return true;
          if (value === false || value === "false") return false;
          throw new Error(`${key} 必须选择开启或关闭`);
        }
        if (key === "freeze") {
          const layers = typeof value === "string" ? JSON.parse(value) : value;
          if (!Number.isInteger(layers) && !(Array.isArray(layers) && layers.every(Number.isInteger))) throw new Error("freeze 请填写整数或层索引数组，如 [0,1,2]");
          return layers;
        }
        const number = Number(value);
        if (!Number.isFinite(number)) throw new Error(`${key} 必须为有限数字`);
        return number;
      },
      /** 参与搜索的维度额外遵守操作者范围，其余使用服务端 Schema。 */
      parameterBound(field, side) {
        const space = this.searchSpace[field.name];
        return side === "min" ? (space?.low ?? field.minimum ?? field.exclusiveMinimum) : (space?.high ?? field.maximum ?? field.exclusiveMaximum);
      },
      parameterOptions(field) {
        const choices = this.searchSpace[field.name];
        const values = Array.isArray(choices) ? [...choices] : [];
        const selected = this.trainForm.params[field.name];
        if (this.trainingReadback && selected !== "" && selected != null && !values.some(value => String(value) === String(selected))) values.push(selected);
        return field.control === "boolean" ? [{ value: "true", label: "开启" }, { value: "false", label: "关闭" }] : values.map(value => ({ value: String(value), label: String(value) }));
      },
      parameterIsSelect(field, agent = true) {
        return field.control === "boolean" || (agent && Array.isArray(this.searchSpace[field.name]));
      },
      /** 固定动作使用用户点击的身份，不接受函数、命令或远程地址。
       * @param {string} path 固定业务路径。
       * @param {string} message 成功后的明确状态。
       * @returns {Promise<void>} 提交结束。
       */
      async action(path, message) {
        if (this.busy) return;
        this.busy = true;
        try {
          await api(path, {
            method: "POST",
            body: /\/(analyses|evaluate)$/.test(path) ? {} : undefined,
          });
          this.toast(message);
          this.busy = false;
          await this.refresh();
        } catch (error) {
          this.error = error.message;
        } finally {
          this.busy = false;
        }
      },
      /** 将操作者主动读取的错误保留在界面，避免异步拒绝仅出现在控制台。
       * @param {Promise<any>} operation 已发起的只读操作。
       * @returns {Promise<void>} 读取结束。
       */
      async read(operation) {
        try {
          await operation;
        } catch (error) {
          this.error = error.message;
        }
      },
      /** Loss 与 mAP 量纲不同，分两张单轴图绘制；重复轮询不播放装饰动画。
       * @returns {void} 无。
       */
      plotTraining() {
        const prefix = this.page === "dashboard" ? "dashboard" : "training";
        const labels = this.epochs.map((row) => row.epoch);
        const series = (key, label, color) => ({
          label,
          data: this.epochs.map((row) => row[key] ?? null),
          borderColor: color,
          backgroundColor: color,
          borderWidth: 2,
          pointRadius: this.epochs.length > 50 ? 0 : 2.5,
          pointHoverRadius: 5,
          tension: 0,
          spanGaps: false,
        });
        const options = (legend, y) => ({
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: {
              display: legend,
              align: "start",
              labels: { usePointStyle: true, pointStyle: "line", boxWidth: 18 },
            },
          },
          scales: {
            x: {
              title: { display: true, text: "epoch" },
              grid: { display: false },
            },
            y: { ...y, ticks: { maxTicksLimit: 5 } },
          },
        });
        const loss = document.getElementById(`${prefix}-loss`);
        const map = document.getElementById(`${prefix}-map`);
        if (loss)
          this.chart(
            `${prefix}-loss`,
            loss,
            {
              labels,
              datasets: [
                series("train/box_loss", "框 Loss", colors[0]),
                series("train/cls_loss", "分类 Loss", colors[1]),
                series("train/l1_loss", "L1 Loss", colors[2]),
              ],
            },
            options(true, {}),
          );
        if (map)
          this.chart(
            `${prefix}-map`,
            map,
            {
              labels,
              datasets: [
                series("metrics/mAP50-95(B)", "mAP50–95", colors[6]),
              ],
            },
            // 保留零基线，避免小幅波动被放大成趋势
            options(false, { min: 0 }),
          );
      },
      /** 曲线和数据数组直接来自验证器，缺少类别支持时不补曲线。
       * @returns {void} 无。
       */
      plotPR() {
        const canvas = document.getElementById("pr-chart");
        const curve = this.evaluationReport?.curves?.find(
          (x) => x.x_label === "Recall" && x.y_label === "Precision",
        );
        if (!canvas || !curve) return;
        const datasets = curve.class_ids.map((id, index) => ({
          label:
            this.evaluationReport.per_class.find((x) => x.class_id === id)
              ?.name || String(id),
          data: curve.x.map((x, j) => ({ x, y: curve.y[index]?.[j] ?? null })),
          borderColor: colors[index % colors.length],
          backgroundColor: colors[index % colors.length],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0,
        }));
        this.chart(
          "pr-chart",
          canvas,
          { datasets },
          {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: "nearest", intersect: false },
            plugins: {
              legend: {
                align: "start",
                labels: {
                  usePointStyle: true,
                  pointStyle: "line",
                  boxWidth: 18,
                },
              },
            },
            scales: {
              x: {
                type: "linear",
                min: 0,
                max: 1,
                title: { display: true, text: "Recall" },
              },
              y: {
                min: 0,
                max: 1,
                title: { display: true, text: "Precision" },
              },
            },
          },
        );
      },
      /** 画布复用之前由成熟图表库管理实例生命周期。
       * @param {string} id 画布身份。
       * @param {HTMLCanvasElement} canvas 真实画布。
       * @param {object} data 实际数据。
       * @param {object} options Chart.js 配置。
       * @returns {void} 无。
       */
      chart(id, canvas, data, options) {
        const old = charts.get(id);
        if (old && old.canvas === canvas) {
          old.data = data;
          old.update("none");
        } else {
          old?.destroy();
          charts.set(id, new Chart(canvas, { type: "line", data, options }));
        }
      },
      /** 切换视图时清理已离开的图表监听器。
       * @returns {void} 无。
       */
      clearCharts() {
        for (const chart of charts.values()) chart.destroy();
        charts.clear();
      },
      /** 短反馈不替代持久任务结果，错误单独保持可见。
       * @param {string} text 操作状态。
       * @returns {void} 无。
       */
      toast(text) {
        this.notice = text;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
          this.notice = "";
        }, 6500);
      },
      /** 缺失值保留为空，真实零值按精度显示。
       * @param {number|null} value 观测值。
       * @param {number} digits 显示精度。
       * @returns {string} 可读指标。
       */
      metric(value, digits = 4) {
        return value == null || !Number.isFinite(Number(value))
          ? "—"
          : Number(value).toFixed(digits);
      },
      /** 单位来自实际字节值，无指标不伪造零显存。
       * @param {number|null} value 字节数。
       * @returns {string} 可读容量。
       */
      bytes(value) {
        return value == null
          ? "—"
          : value >= 1024 ** 3
            ? `${(value / 1024 ** 3).toFixed(1)} GiB`
            : `${(value / 1024 ** 2).toFixed(1)} MiB`;
      },
      /** 显示观测时间而非猜测任务仍在线。
       * @param {string|null} value ISO 时间。
       * @returns {string} 本地时间。
       */
      time(value) {
        return value
          ? new Date(value).toLocaleString("zh-CN", { hour12: false })
          : "—";
      },
      /** 动态列表只需要时分，跨天时补上日期。
       * @param {string|null} value ISO 时间。
       * @returns {string} 简短时间。
       */
      clock(value) {
        if (!value) return "—";
        const date = new Date(value);
        return date.toDateString() === new Date().toDateString()
          ? date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit" })
          : date.toLocaleString("zh-CN", { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
      },
      /** 标识缩略只用于显示，所有请求仍使用完整 ID。
       * @param {string} value 业务标识。
       * @returns {string} 显示文本。
       */
      short(value) {
        return value ? value.slice(0, 8) : "—";
      },
      /** 封闭状态映射避免把后端文本当作样式或 HTML。
       * @param {string} value 真实状态。
       * @returns {string} 中文状态。
       */
      status(value) {
        return labels[value] || value || "—";
      },
      /** 阶段名只转译真实节点，不创造完成记录。
       * @param {string} value 后端阶段。
       * @returns {string} 中文阶段。
       */
      phase(value) {
        return phases[value] || value || "等待开始";
      },
      /** 颜色只表达已知状态，文字同时表达结果。
       * @param {string} value 状态。
       * @returns {string} 固定 CSS 类。
       */
      tone(value) {
        return [
          "completed",
          "reviewed",
          "exported",
          "valid",
          "published",
        ].includes(value)
          ? "success"
          : ["failed", "corrupt"].includes(value)
            ? "danger"
            : [
                  "needs_review",
                  "pending_review",
                  "paused",
                  "canceling",
                ].includes(value)
              ? "warning"
              : ACTIVE.includes(value)
                ? "info"
                : "neutral";
      },
      /** 只允许明确的 HTTP 审核/追踪链接。
       * @param {string} value 操作者配置的服务地址。
       * @returns {string} 安全链接或空锚点。
       */
      url(value) {
        return /^https?:\/\//i.test(value || "") ? value : "#";
      },
      /** 预览框沿原图比例叠加，编辑仍由 CVAT 完成。
       * @param {object} box 像素 xyxy 标注。
       * @returns {object} 百分比样式。
       */
      boxStyle(box) {
        const image = this.annotationImage?.image;
        return image
          ? {
              left: `${(box.bbox[0] / image.width) * 100}%`,
              top: `${(box.bbox[1] / image.height) * 100}%`,
              width: `${((box.bbox[2] - box.bbox[0]) / image.width) * 100}%`,
              height: `${((box.bbox[3] - box.bbox[1]) / image.height) * 100}%`,
            }
          : {};
      },
      /** 评估器输出归一化左上角与宽高，叠框不改变图片纵横比；TP/FP/FN 颜色由样式表按状态类决定。
       * @param {object} box 验证集匹配框。
       * @returns {object} 百分比布局。
       */
      errorBoxStyle(box) {
        return {
          left: `${box.bbox[0] * 100}%`,
          top: `${box.bbox[1] * 100}%`,
          width: `${box.bbox[2] * 100}%`,
          height: `${box.bbox[3] * 100}%`,
        };
      },
      /** 主流程某一步相对当前流程的状态。
       * @param {number} step 1 到 5。
       * @returns {string} done、current、gate、todo、skipped 或 failed。
       */
      stepState(step) {
        const run = this.agent;
        if (!run) return step === 1 ? "current" : "todo";
        if (this.reusedExport && step <= 3) return "skipped";
        const at = this.flowStep;
        if (step < at) return "done";
        if (step > at) return "todo";
        if (run.status === "failed") return "failed";
        if (run.status === "needs_review") return "gate";
        return "current";
      },
      /** 步骤的简短状态文字，颜色之外始终有文字说明。
       * @param {number} step 1 到 5。
       * @returns {string} 状态文字。
       */
      stepText(step) {
        const state = this.stepState(step);
        if (step === 5 && state === "current" && ["completed", "stopped"].includes(this.agent?.status))
          return "可以导出";
        return {
          done: "已完成",
          current: this.agent ? "进行中" : "从这里开始",
          gate: "等你处理",
          todo: "未开始",
          skipped: "复用已审数据",
          failed: "失败",
        }[state];
      },
      /** 流程条的样式类：当前与等待人工的一步带检测框角标。
       * @param {number} step 1 到 5。
       * @param {string} id 对应视图。
       * @returns {string} 类名。
       */
      stepClass(step, id) {
        const state = this.stepState(step);
        return `is-${state}${["current", "gate"].includes(state) ? " det" : ""}${this.page === id ? " selected" : ""}`;
      },
      /** 后台任务类型的中文名。
       * @param {string} kind 任务类型。
       * @returns {string} 中文名。
       */
      taskName(kind) {
        return (
          {
            reviewed_import: "已审核数据导入",
            labeled_import: "已标注数据导入",
            dataset_export: "数据集导出",
            import: "图片导入",
            analysis: "质量分析",
            version: "版本封存",
            auto_annotate: "候选推理",
            cvat_push: "发送 CVAT",
            cvat_pull: "回传人工审核",
            yolo_export: "封存训练数据",
          }[kind] || kind
        );
      },
      /** 部署格式的显示名。
       * @param {string} format 格式标识。
       * @returns {string} 显示名。
       */
      formatName(format) {
        return { onnx: "ONNX", torchscript: "TorchScript" }[format] || format;
      },
      /** 某个权重的导出记录。
       * @param {string} modelId 注册权重。
       * @returns {object[]} 最新在前的导出。
       */
      exportsFor(modelId) {
        return this.modelExports.filter((x) => x.model_id === modelId);
      },
      /** 参数片段转成紧凑文字。
       * @param {object} values 参数。
       * @returns {string} 例如 epochs=4 · lr0=0.005。
       */
      paramText(values) {
        return Object.entries(values || {})
          .map(([key, value]) => `${key}=${value}`)
          .join(" · ");
      },
      /** 工具调用的关键参数，理由单独展示。
       * @param {string} tool 工具名。
       * @param {object} args 调用参数。
       * @returns {string} 摘要。
       */
      toolDetail(tool, args) {
        if (tool === "start_experiment")
          return this.paramText(args.parameters) || "沿用上一轮参数";
        if (tool === "read_chart")
          return `${args.source === "training" ? "训练" : "评估"} · ${(args.charts || []).map((x) => chartNames[x] || x).join("、")}`;
        if (tool === "sleep") return `${args.minutes} 分钟`;
        if (tool === "export_model")
          return `${this.formatName(args.format)} · ${args.kind === "last" ? "最后轮" : "最佳轮"}权重`;
        if (tool === "finish") return args.summary || "";
        if (tool === "request_human_review") return (args.data_actions || []).join("；");
        return args.experiment_id ? `实验 ${this.short(args.experiment_id)}` : "";
      },
      /** 每轮实验相对上一轮的参数变化。
       * @param {object} item 迭代记录。
       * @returns {string} 变化摘要。
       */
      changeSummary(item) {
        const design = item.conclusion?.comparison || item.plan?.comparison;
        if (design?.parent_experiment_id) {
          const mode = item.plan?.comparison_mode === "ablation" ? "定向消融" : "联合搜索";
          const changes = Object.entries(design.changed_variables || {})
            .map(([key, value]) => `${key} ${JSON.stringify(value.before)}→${JSON.stringify(value.after)}`);
          return `${mode} · ${changes.join(" · ") || "参数不变"}`;
        }
        const index = this.agentIterations.indexOf(item);
        const before = this.agentIterations[index - 1]?.plan?.parameters;
        const now = item.plan?.parameters || {};
        if (!before) return `首轮 · ${now.model} · ${now.epochs} epochs · ${now.imgsz}`;
        const changes = Object.entries(now)
          .filter(([key, value]) => before[key] !== value)
          .map(([key, value]) => `${key} ${before[key]}→${value}`);
        if (item.plan?.dataset_export_id && item.plan.dataset_export_id !== this.agentIterations[index - 1].plan?.dataset_export_id)
          changes.push(`数据 → ${this.short(item.plan.dataset_export_id)}`);
        return `旧记录相对上一轮 · ${changes.join(" · ") || "参数不变"}`;
      },
      /** 执行事件的中文名。
       * @param {object} event 运行事件。
       * @returns {string} 中文名。
       */
      eventName(event) {
        if (event.event_type === "tool_call")
          return `调用：${tools[event.payload.tool] || event.payload.tool}`;
        return (
          {
            entered: "进入阶段",
            created: "创建流程",
            model_decision: "模型决策",
            model_response_received: "模型响应",
            woke: "醒来",
            validation_observed: "验证结论",
            experiment_created: "新建实验",
            experiment_abandoned: "放弃实验",
            checkpoint_resumed: "从检查点继续",
            training_stop_requested: "请求暂停训练",
            stagnation_detected: "停滞停止",
            human_review_requested: "请求人工处理",
            training_setup: "开始训练",
            sealed: "报告封存",
            failed: "失败",
            pause: "暂停",
            resume: "恢复",
            stop: "停止",
            paused: "已暂停",
          }[event.event_type] || event.event_type
        );
      },
      get globalNav() {
        return this.nav.filter(item => item.scope === "global");
      },
      get projectNav() {
        return this.nav.filter(item => item.id === "dashboard").concat(this.nav.filter(item => item.scope !== "global" && item.id !== "dashboard"));
      },
      get isProjectView() {
        return this.projectNav.some(item => item.id === this.page);
      },
      get currentProject() {
        return this.projects.find(project => project.id === this.projectId) || null;
      },
      projectName(id) {
        return this.projects.find(project => project.id === id)?.name || this.short(id);
      },
      get pageTitle() {
        return this.nav.find((x) => x.id === this.page)?.name || "项目列表";
      },
      get pageEyebrow() {
        const item = this.nav.find((x) => x.id === this.page);
        return item?.step ? `步骤 ${item.step} / 5 · ${this.currentProject?.name || ''}` : this.isProjectView ? this.currentProject?.name || "项目空间" : "所有项目";
      },
      get pageSub() {
        if (this.page === "dashboard") return this.overview?.project?.name || "";
        if (this.page === "projects") return "选择一个项目，进入它的数据、标注、训练和模型工作区";
        if (this.page === "overview") return "汇总所有项目的运行状态和数据规模";
        if (this.page === "upload") return "上传原图或已标注数据集，管理数据版本与导出";
        if (!this.flowPage) return "";
        if (!this.agent) return "还没有流程，从第 1 步开始";
        return this.agent.requirement ? `需求：${this.agent.requirement}` : this.agent.name;
      },
      get flowPage() {
        return ["dashboard", "upload", "annotate", "review", "train", "export"].includes(this.page);
      },
      get currentDataset() {
        return this.datasets.find((x) => x.id === this.datasetId);
      },
      get currentAnnotation() {
        return this.annotations.find((x) => x.id === this.annotationId);
      },
      get cvatEmbedUrl() {
        const task = this.currentAnnotation?.cvat_task_id;
        if (!/^[1-9]\d*$/.test(String(task || ""))) return "";
        try {
          const target = new URL(this.overview?.configuration?.cvat_url);
          if (!["http:", "https:"].includes(target.protocol) || target.username || target.password) return "";
          // 同机保持同一站点，使 CVAT 正常使用自身的 SameSite 登录 Cookie。
          const loopback = ["localhost", "127.0.0.1"];
          if (loopback.includes(target.hostname) && loopback.includes(location.hostname)) target.hostname = location.hostname;
          if (location.protocol === "https:" && target.protocol === "http:") return "";
          target.pathname = `${target.pathname.replace(/\/$/, "")}/tasks/${task}`;
          target.search = target.hash = "";
          return target.href;
        } catch (_) { return ""; }
      },
      get worker() {
        return this.runtime?.latest_worker?.runtime || null;
      },
      get limits() {
        return this.overview?.configuration?.upload_limits || {};
      },
      get llmReady() {
        return Boolean(this.overview?.configuration?.llm_configured);
      },
      get searchSpace() {
        return this.overview?.configuration?.search_space || {};
      },
      get trainingParameterGroups() {
        return (this.overview?.configuration?.training_parameter_groups || []).map(group => ({ ...group, fields: group.fields.filter(field => field.name !== "lr0") }));
      },
      get trainingOptimizers() {
        return this.overview?.configuration?.training_optimizers || this.searchSpace.optimizer || [];
      },
      get preparedModels() {
        return this.overview?.configuration?.prepared_models || [];
      },
      get imgszChoices() {
        const choices = (this.searchSpace.imgsz || []).filter((x) => x <= Number(this.trainForm.budget.max_imgsz || Infinity));
        const selected = this.trainForm.params.imgsz;
        return this.trainingReadback && selected && !choices.includes(Number(selected)) ? [...choices, Number(selected)] : choices;
      },
      get trainingModelChoices() {
        return [...new Set([...this.preparedModels, ...(this.trainingReadback && this.trainForm.params.model ? [this.trainForm.params.model] : [])])];
      },
      get trainingOptimizerChoices() {
        return [...new Set([...(this.searchSpace.optimizer || this.trainingOptimizers), ...(this.trainingReadback && this.trainForm.params.optimizer ? [this.trainForm.params.optimizer] : [])])];
      },
      get dialogTitle() {
        return (
          {
            project: "创建项目",
            dataset: "新建数据集",
            version: "封存分组版本",
            schema: "新建类别版本",
            annotate: "生成候选标注",
            sync: "回传人工审核",
            experiment: "手动创建实验",
            resumeAgent: this.agent?.phase === "cvat_review" ? "确认 CVAT 审核完成" : "人工处理与恢复",
            publish: "确认生产发布",
            annotationImage: "标注只读预览",
          }[this.modal] || ""
        );
      },
      get focusRun() {
        return (
          this.agents.find(
            (x) => ACTIVE.includes(x.status) || x.status === "needs_review",
          ) ||
          this.agents.find(
            (x) => x.status === "completed" && x.best_model_id,
          ) ||
          this.agents[0] ||
          null
        );
      },
      get flowStep() {
        const run = this.agent;
        if (!run) return 1;
        if (["completed", "stopped"].includes(run.status)) return 5;
        return PHASE_STEP[run.phase] || 2;
      },
      get reusedExport() {
        // 从已审数据直接训练的流程没有自己的标注批次。
        return Boolean(this.agent?.dataset_export_id && !this.agent?.state?.annotation_run_id);
      },
      get setupMode() {
        return this.agent?.phase === "training_setup" && this.agent?.status === "needs_review";
      },
      get showTrainForm() {
        return this.setupMode || this.trainNew || !this.agent;
      },
      get trainingReadback() {
        return !this.showTrainForm && Boolean(this.agent && this.flowStep >= 4);
      },
      get latestTrainingIteration() {
        return this.agentIterations.filter(item => item.experiment_id && item.plan?.parameters)
          .reduce((latest, item) => !latest || item.iteration > latest.iteration ? item : latest, null);
      },
      get trainingParameterSource() {
        const iteration = this.latestTrainingIteration;
        return iteration ? `第 ${iteration.iteration + 1} 轮 · 实验 ${this.short(iteration.experiment_id)} · Agent 已选定` : "等待 Agent 规划首轮参数；已填写值保留显示";
      },
      get classRows() {
        return this.pipeline.classes.filter((row) => row.name.trim() || row.prompt.trim());
      },
      get startBlocker() {
        if (!this.datasetId) return "先选择或新建数据集，并上传图片";
        if (!this.images.length) return "数据集里还没有图片";
        if (!this.pipeline.schemaId) {
          const rows = this.classRows;
          if (!rows.length) return "至少需要一个类别：名称和英文提示词";
          if (rows.some((row) => !row.name.trim() || !PROMPT.test(row.prompt.trim())))
            return "每个类别都要有名称；英文提示词以字母开头，只含字母、数字、空格、逗号、连字符或撇号";
        }
        if (!this.llmReady) return "未配置模型服务，无法启动自主流程";
        return "";
      },
      get agentStateText() {
        const run = this.agent;
        if (!run) return "";
        const waiting = run.state?.waiting || {};
        if (run.status === "failed") return "运行失败，等待人工恢复";
        if (run.status === "paused") return "已暂停";
        if (run.status === "canceling") return "正在停止";
        if (run.status === "stopped") return "已停止";
        if (run.status === "completed") return "已完成，可以导出模型";
        if (run.status === "needs_review")
          return (
            {
              cvat_review: "等你在 CVAT 审核",
              training_setup: "等你确认参数并开始训练",
              agent_review: "Agent 请求你处理",
            }[run.phase] || "等待人工处理"
          );
        if (run.phase === "agent_sleep" && waiting.kind === "sleep")
          return `休眠中，预计 ${this.clock(waiting.until)} 醒来`;
        return this.phase(run.phase);
      },
      get agentStateDetail() {
        const run = this.agent;
        if (!run) return "";
        const waiting = run.state?.waiting || {};
        if (run.phase === "agent_sleep" && waiting.kind === "sleep")
          return `${waiting.reason}。训练、评估或导出结束时会提前醒来。`;
        if (["completed", "stopped"].includes(run.status))
          return run.state?.stop_reason || "";
        return this.agentEvents.findLast((x) => x.event_type === "model_decision" && x.payload.content)?.payload.content || "";
      },
      get flowTone() {
        const status = this.agent?.status;
        return status === "needs_review" || status === "paused"
          ? "warn"
          : status === "failed"
            ? "bad"
            : status === "completed"
              ? "ok"
              : ACTIVE.includes(status)
                ? "live"
                : "muted";
      },
      get pipelineStages() {
        const run = this.agent;
        if (!run) return [];
        const at = PHASE_ORDER.indexOf(run.phase);
        const finished = ["completed", "stopped"].includes(run.status);
        const batch = this.annotations.find((x) => x.id === run.state?.annotation_run_id);
        const analysis = run.state?.dataset?.analysis;
        const stages = [
          { key: "inspect", name: "检查数据", phases: ["dataset_inspector", "data_quality_check"], detail: analysis ? `有效图片 ${analysis.valid_images} 张` : "" },
          { key: "split", name: "划分 train / val / test 并封存", phases: ["dataset_split"], detail: run.dataset_version_id ? `版本 ${this.short(run.dataset_version_id)}` : "" },
          {
            key: "annotate",
            name: "Grounding DINO 生成候选框",
            phases: ["auto_annotation", "annotation_quality"],
            detail: batch ? `${batch.processed} / ${batch.total} 张图` : "",
            done: batch?.processed,
            total: batch && ["queued", "running"].includes(batch.status) ? batch.total : 0,
          },
          { key: "cvat", name: "发送到 CVAT", phases: ["cvat_upload"], detail: batch?.cvat_task_id ? `CVAT 任务 #${batch.cvat_task_id}` : "" },
        ];
        return stages.map((stage) => {
          const first = PHASE_ORDER.indexOf(stage.phases[0]);
          const last = PHASE_ORDER.indexOf(stage.phases.at(-1));
          const state = this.reusedExport
            ? "skipped"
            : finished || at > last
              ? "done"
              : at >= first
                ? run.status === "failed"
                  ? "failed"
                  : "current"
                : "todo";
          return { ...stage, state, label: { done: "完成", current: "进行中", failed: "失败", todo: "等待", skipped: "已复用" }[state] };
        });
      },
      get timeline() {
        const items = [];
        for (const event of this.agentEvents) {
          const p = event.payload || {};
          let item = null;
          if (event.event_type === "tool_call") {
            const args = p.arguments && typeof p.arguments === "object" ? p.arguments : {};
            let error = "";
            if (!p.ok)
              try {
                error = JSON.parse(p.result).error;
              } catch (_) {
                error = p.result;
              }
            item = {
              title: tools[p.tool] || p.tool,
              detail: p.ok ? this.toolDetail(p.tool, args) : `被拒绝：${error}`,
              reason: args.reason,
              tone: p.ok ? (p.tool === "sleep" ? "muted" : "info") : "danger",
            };
          } else if (event.event_type === "model_decision" && p.content)
            item = { title: "Agent 说明", detail: p.content, tone: "muted" };
          else if (event.event_type === "woke") item = { title: "醒来", detail: p.reason, tone: "muted" };
          else if (event.event_type === "validation_observed")
            item = {
              title: `第 ${p.iteration + 1} 轮验证完成`,
              detail: `mAP50–95 ${this.metric(p.summary?.map50_95)} · mAP50 ${this.metric(p.summary?.map50)}`,
              tone: "success",
            };
          else if (event.event_type === "experiment_abandoned")
            item = { title: `第 ${p.iteration + 1} 轮已放弃`, detail: this.status(p.status), tone: "warning" };
          else if (event.event_type === "stagnation_detected")
            item = { title: "连续无明显提升，停止训练", detail: p.message, tone: "warning" };
          else if (event.event_type === "human_review_requested")
            item = { title: "请求人工处理", detail: p.reason, tone: "warning" };
          else if (event.event_type === "training_setup")
            item = { title: "操作者开始训练", detail: this.paramText(p.initial_parameters) || "首轮参数全部交给 Agent", tone: "info" };
          else if (event.event_type === "failed")
            item = { title: `${this.phase(event.node)}失败`, detail: p.message, tone: "danger" };
          else if (event.event_type === "sealed")
            item = { title: "运行报告已封存", detail: p.best_model ? `最佳权重 ${this.short(p.best_model.model_id)}` : "没有有效候选", tone: "success" };
          else if (event.node === "operator" && ["pause", "resume", "stop"].includes(event.event_type))
            item = { title: { pause: "操作者暂停", resume: "操作者恢复", stop: "操作者停止" }[event.event_type], detail: p.reason || "", tone: "muted" };
          if (item) items.push({ id: event.id, time: event.created_at, ...item });
        }
        return items.reverse();
      },
      get pendingActions() {
        const items = [];
        for (const run of this.agents) {
          if (run.status === "needs_review")
            items.push({
              run,
              title: { cvat_review: "在 CVAT 完成审核", training_setup: "确认参数并开始训练", agent_review: "Agent 请求你处理" }[run.phase] || "等待人工处理",
              page: { cvat_review: "review", training_setup: "train", agent_review: "train" }[run.phase] || "agents",
              tone: "gate",
            });
          else if (run.status === "failed")
            items.push({ run, title: "运行失败，需要人工恢复", page: "train", tone: "danger", detail: run.error });
        }
        for (const notice of this.overview?.notices || [])
          items.push({ run: { id: notice.run_id, name: notice.name }, title: "已因连续无明显提升停止", page: "train", tone: "warning", detail: notice.notice.message });
        return items;
      },
      get flowBest() {
        let best = null;
        for (const item of this.agentIterations) {
          const value = item.conclusion?.map50_95;
          if (value != null && (!best || value > best.map50_95))
            best = { iteration: item.iteration, experiment_id: item.experiment_id, map50_95: value };
        }
        return best;
      },
      get exportCandidate() {
        const run = this.agent;
        let model = run?.best_model_id ? this.models.find((x) => x.id === run.best_model_id) : null;
        if (!model && this.flowBest)
          model = this.models.find((x) => x.experiment_id === this.flowBest.experiment_id && x.kind === "best");
        if (!model) return null;
        const iteration = this.agentIterations.find((x) => x.experiment_id === model.experiment_id);
        return {
          model_id: model.id,
          experiment_id: model.experiment_id,
          kind: model.kind,
          sha256: model.sha256,
          iteration: iteration?.iteration ?? 0,
          map50_95: iteration?.conclusion?.map50_95 ?? null,
          source: run?.best_model_id === model.id ? "Agent 选定的最佳模型" : "当前最佳候选",
        };
      },
    };
  }
  document.addEventListener("alpine:init", () => {
    // 图表字体与配色跟随界面，避免 Chart.js 默认的灰色小字
    Object.assign(Chart.defaults, {
      color: ink.text,
      borderColor: ink.grid,
    });
    Chart.defaults.font.family = getComputedStyle(document.documentElement)
      .getPropertyValue("--sans")
      .replace(/\s+/g, " ")
      .trim();
    Chart.defaults.font.size = 12;
    Alpine.data("lab", lab);
  });
})();
