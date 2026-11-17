/* 业务状态由 Alpine 管理；图表复用 Chart.js，不把缺失指标填成零。 */
(() => {
  const API = "/api/v1";
  const ACTIVE = ["queued", "running", "waiting", "canceling"];
  const labels = {
    created: "已创建",
    uploading: "上传中",
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
    Skill: "读取训练技能",
    SkillRead: "读取技能参考",
    inspect_training_parameters: "查询可调参数",
    read_dataset_examples: "检查已审样例",
    read_validation_errors: "筛选验证错误",
    compare_experiments: "对比历史实验",
    preview_experiment: "预检训练方案",
    answer_operator: "回答你的问题",
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
    datasetCatalog: "/workspace/datasets",
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
    const uploadRequests = new Map();
    let promptOptimizationRequest = null;
    let promptOptimizationTicket = 0;
    let agentChatTicket = 0;
    let trainingLogResumeTimer = null;
    let trainingLogPointer = null;
    let trainingLogTouching = false;
    let trainingLogView = 0;
    let uploadSequence = 0;
    let labeledFiles = [];
    const detailTickets = {
      dataset: 0,
      annotation: 0,
      cvat: 0,
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
      datasetCatalog: [],
      scopeVersionId: "",
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
      copySources: [],
      copyVersions: [],
      copyVersionsLoading: false,
      copySourceTicket: 0,
      imageOffset: 0,
      imagesMore: false,
      uploadCount: 0,
      uploads: [],
      uploadMeta: "{}",
      uploadMode: "images",
      labeledUpload: { name: "", count: 0, bytes: 0, reviewed: true, reviewer: "上传者", confirmation: "上传者声明该数据已完成标注、逐图审核和漏标检查", progress: null },
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
      preparationFocus: "",
      annotationPinned: false,
      annotationQuality: null,
      annotationTasks: [],
      annotationImages: [],
      annotationImageOffset: 0,
      annotationImageKey: "",
      annotationQualityKey: "",
      annotationImage: null,
      cvatSessionUrl: "",
      cvatSessionKey: "",
      cvatAttemptKey: "",
      cvatConnecting: false,
      cvatError: "",
      experimentId: "",
      experimentPinned: false,
      experiment: null,
      trainingTasks: [],
      metrics: [],
      epochs: [],
      trainingLogs: "",
      trainingLogAutoFollow: true,
      trainingTaskId: "",
      trainingCharts: [],
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
      viewer: null,
      agentId: "",
      agent: null,
      agentEvents: [],
      operatorMessages: [],
      operatorInstructions: [],
      agentChatOpen: false,
      agentQuestion: "",
      agentWakeSending: false,
      agentWakeError: "",
      agentWakeAcceptedUntil: "",
      agentIterations: [],
      agentBudget: null,
      eventCursor: 0,
      trainForm: { params: emptyParams(), budget: {}, exportId: "", requirement: "", name: "", lockModel: false },
      trainFormKey: "",
      trainNew: false,
      promptOptimizer: { busy: false, error: "", draft: "", notes: "", context: "", original: "", appliedText: "" },
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
        window.addEventListener("beforeunload", event => {
          if (this.uploads.some(item => ["preparing", "uploading", "receiving", "canceling"].includes(item.phase))) {
            event.preventDefault();
            event.returnValue = "";
          }
        });
        document.addEventListener("visibilitychange", () => {
          if (!document.hidden) this.refresh();
        });
        await this.refresh(true);
        const tick = async () => {
          if (!document.hidden) await this.pollUploads();
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
        this.resetTrainingLogFollow();
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
        if (next.page !== this.page) {
          this.resetTrainingLogFollow();
          ++detailTickets.cvat;
          this.cvatSessionUrl = this.cvatSessionKey = this.cvatAttemptKey = this.cvatError = "";
          this.cvatConnecting = false;
        }
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
        this.resetPromptOptimization();
        this.resetAgentChat();
        this.resetTrainingLogFollow();
        this.cvatSessionUrl = this.cvatSessionKey = this.cvatAttemptKey = this.cvatError = "";
        this.cvatConnecting = false;
        labeledFiles = [];
        this.labeledUpload = { name: "", count: 0, bytes: 0, reviewed: true, reviewer: "上传者", confirmation: "上传者声明该数据已完成标注、逐图审核和漏标检查", progress: null };
        this.scopeVersionId = "";
        this.preparationFocus = "";
        this.annotationImageOffset = 0; this.annotationImageKey = this.annotationQualityKey = "";
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
        this.evaluations = []; this.modelExports = [];
        this.epochs = [];
        this.images = [];
        this.annotationImages = [];
        this.annotationTasks = [];
        this.datasetTasks = [];
        this.datasetVersions = [];
        this.trainingTasks = [];
        this.trainingCharts = [];
        this.uploadCount = 0;
        if (this.$refs.uploadFiles) this.$refs.uploadFiles.value = "";
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
              if (key === "datasetCatalog") { this[key] = fresh; return; }
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
        if (["upload", "annotate", "review"].includes(this.page)) {
          if (!this.datasetId) this.datasetId = this.datasets[0]?.id || "";
          if (this.datasetId) await this.chooseDataset(this.datasetId, false);
        }
        if (!current()) return;
        if (["annotate", "review"].includes(this.page)) {
          // 默认跟随当前流程自己的标注批次，手动选过其它批次后不再被轮询覆盖。
          this.annotationId = this.scopeVersion?.annotation_run_id || "";
          if (this.annotationId)
            await this.chooseAnnotation(this.annotationId, false);
          else { this.annotationQuality = null; this.annotationTasks = []; this.annotationImages = []; }
          if (current()) this.focusPreparationSection();
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
        const versions = this.datasetCatalog.find(row => row.id === id)?.versions || [];
        if (!versions.some(row => row.id === this.scopeVersionId)) this.scopeVersionId = versions[0]?.id || "";
        this.imagesMore = rows[0].length === 24;
      },
      get scopeDataset() { return this.datasetCatalog.find(row => row.id === this.datasetId) || null; },
      get scopeVersion() { return this.scopeDataset?.versions.find(row => row.id === this.scopeVersionId) || null; },
      get preparationAgent() {
        const run = this.agent;
        if (!run || !(run.state?.workflow_mode === 'preparation' || run.state?.annotation_run_id)) return null;
        return run.dataset_id === this.datasetId && (!this.scopeVersion || !run.dataset_version_id || run.dataset_version_id === this.scopeVersion.id) ? run : null;
      },
      get annotationFlowAgent() {
        const run = this.preparationAgent;
        return run && (!this.currentAnnotation || run.state?.annotation_run_id === this.annotationId) ? run : null;
      },
      get preparationTask() {
        return this.currentAnnotation ? this.annotationTasks.find(task => task.parameters?.run_id === this.annotationId && ["queued", "running", "canceling"].includes(task.status)) || null : null;
      },
      /** 所有提示和主操作围绕所选数据版本，不把其它流程的状态用于当前批次。 */
      get preparationGuide() {
        const batch = this.currentAnnotation, run = this.annotationFlowAgent, task = this.preparationTask;
        const exported = batch ? this.exports.find(item => item.run_id === batch.id) : null;
        const scopedExport = !batch || batch.id === this.scopeVersion?.annotation_run_id ? this.scopeVersion?.dataset_export_id : null;
        const exportId = exported?.id || scopedExport || null;
        const step = batch?.status === "reviewed" || run?.phase === "reviewed_export" ? 3 : run?.phase === "annotation_validation" ? 2 : (batch?.cvat_task_id && batch.status === "pending_review") || run?.phase === "cvat_review" ? 1 : 0;
        const base = { key: "empty", step, owner: "从这里开始", tone: "info", action: "upload", label: "去上传与需求", secondary: "", secondaryLabel: "", exportId };
        if (exportId || batch?.status === "exported") return { ...base, key: "ready", step: 4, owner: "数据已就绪", tone: "success", title: "审核和封存已完成，可以选择这份数据训练", description: "已保存审核结果和固定的数据划分。点击后前往训练页填写需求、确认参数与预算，再开始训练。", action: "train", label: "选择这份数据训练 →" };
        if (task?.status === "canceling" || run?.status === "canceling" || run?.stop_requested) return { ...base, key: "canceling", owner: "系统正在停止", title: "正在结束当前处理，请稍候", description: "停止完成后会保留已有标签和处理记录。此时请勿重复提交自动标注、回传或封存。", action: "", label: "" };
        if (run?.status === "paused") return { ...base, key: "paused", owner: "流程已暂停", tone: "warning", title: "流程已暂停，恢复后继续当前步骤", description: "已有候选框和审核记录会保留。恢复时请填写操作者与说明；如果正在审核，请先保存 CVAT 中的修改。", action: "resume", label: "恢复当前流程" };
        if (batch?.status === "stopped" || run?.status === "stopped") return { ...base, key: "stopped", owner: "流程已停止", tone: "warning", title: "这次处理已停止，已有标签保留", description: "可以继续查看已有标注与任务记录。需要重新处理时，请回到上传与需求页选择数据并创建新流程。", action: "upload", label: "返回上传与需求" };
        const latestTask = batch ? this.annotationTasks.find(item => item.parameters?.run_id === batch.id) : null;
        const failedTask = !run && latestTask?.status === "failed" && !task;
        if (run?.status === "failed" || batch?.status === "failed" || failedTask) {
          const retry = !run && ["pending_review", "reviewed"].includes(batch?.status)
            ? ({ cvat_push: ["send", "重试发送到 CVAT"], cvat_pull: ["confirm", "修正后重新确认回传"], yolo_export: ["seal", "重试封存训练数据"] })[latestTask?.kind] : null;
          return { ...base, key: "failed", owner: "需要处理", tone: "danger", title: "当前步骤未完成，请先查看失败原因", description: "已有标签不会被当作审核完成。先根据任务中的错误处理问题，再重试；不需要重复上传已存在的原图。", action: run?.status === "failed" ? "resume" : retry?.[0] || "tasks", label: run?.status === "failed" ? "处理问题并重试" : retry?.[1] || "查看任务与失败原因", secondary: run?.status === "failed" || retry ? "tasks" : "", secondaryLabel: "查看失败原因" };
        }
        if (task?.kind === "cvat_pull" || run?.phase === "annotation_validation" || (run?.phase === "cvat_review" && run.review_confirmation && ["queued", "running"].includes(run.status))) return { ...base, key: "syncing", step: 2, owner: "系统正在核验", title: "正在从 CVAT 回传并核验审核结果", description: "系统会检查每个作业的验收状态、类别与实际标注框。此时无需重复点击回传；可离开页面，稍后回来查看结果。", action: "", label: "" };
        if (task?.kind === "yolo_export" || run?.phase === "reviewed_export") return { ...base, key: "sealing", step: 3, owner: "系统正在封存", title: "审核已通过，正在准备可训练的数据", description: "正在核验图片、标注与固定划分。完成后会显示“选择这份数据训练”，点击后再设置训练需求和预算。", action: "", label: "" };
        if (batch?.status === "reviewed") return { ...base, key: "reviewed", step: 3, owner: "还差最后一步", title: "审核已通过，请封存为训练数据", description: this.importedAnnotation && !batch.cvat_task_id ? "来源审核记录已保留，尚需生成可供训练选择的数据。封存会保留已有标签与固定划分，不会立即启动训练。" : "回传已完成，但还没有生成可供训练选择的数据。封存会保留这次审核结果和固定划分，不会立即启动训练。", action: "seal", label: "封存为训练数据" };
        if (task?.kind === "cvat_push" || run?.phase === "cvat_upload") return { ...base, key: "uploading", owner: "系统正在准备审核", title: "候选框已生成，正在送入 CVAT", description: "正在上传图片和候选标注。工作区准备好后会显示进入审核的按钮，现在不需要手动重复发送。", action: "", label: "" };
        if (batch?.status === "pending_review" && batch.cvat_task_id) return { ...base, key: "review", step: 1, owner: "现在需要你审核", tone: "warning", title: "候选框已准备好，请逐图检查后回传", description: "在 CVAT 修正框、类别和漏标，保存修改，并将所有作业设为验收完成。完成后回到本页，点击“确认审核并回传”。", action: "review", label: "进入 CVAT 逐图审核", secondary: "confirm", secondaryLabel: "确认审核并回传" };
        if (batch?.status === "pending_review") return { ...base, key: "send", owner: "下一步：准备人工审核", title: this.importedAnnotation ? "标签已导入，请发送到 CVAT 审核" : "候选框已生成，请发送到 CVAT 审核", description: "发送会创建审核任务并上传图片与标签。之后还需逐图检查，包括没有框的图片；导入或自动生成都不等于审核通过。", action: "send", label: "发送到 CVAT，准备审核" };
        if (batch || run) return { ...base, key: "annotating", owner: "系统正在处理", title: "正在检查数据并生成待审候选框", description: run ? "系统依次检查图片、固定数据划分、生成候选框并准备审核工作区。你现在可以等待或查看进度；候选框完成后仍需人工审核。" : "正在为所选版本生成候选框。完成后请点击“发送到 CVAT，准备审核”，再逐图检查；不会自动批准候选框。", action: "tasks", label: "查看处理进度" };
        return { ...base, title: "先选择数据，再开始自动标注或标签审核", description: "原图：在上传与需求页填写检测目标并开始自动标注。已有标签：上传标注数据后，根据是否已完成审核进入相应步骤。" };
      },
      get preparationMilestones() {
        const step = this.preparationGuide.step;
        return [
          { title: this.importedAnnotation ? "导入标签" : "生成候选框", owner: "系统处理" },
          { title: this.importedAnnotation && !this.currentAnnotation?.cvat_task_id && step >= 3 ? "继承来源审核" : "在 CVAT 逐图审核", owner: "你来检查" },
          { title: this.importedAnnotation && !this.currentAnnotation?.cvat_task_id && step >= 3 ? "核验已有标签" : "回传并核验", owner: "你确认，系统核验" },
          { title: "封存为训练数据", owner: "完成后手动开训" },
        ].map((item, index) => ({ ...item, state: step > index ? "done" : step === index ? "current" : "todo" }));
      },
      async preparationAction(action) {
        if (this.busy) return;
        if (action === "upload") return this.navigate("upload");
        if (action === "train") return this.preparationGuide.exportId ? this.useReadyDataset(this.preparationGuide.exportId) : this.navigate("train");
        if (action === "resume") return this.show("resumeAgent");
        if (action === "confirm") return this.show(this.annotationFlowAgent?.phase === "cvat_review" && this.annotationFlowAgent.status === "needs_review" ? "resumeAgent" : "sync");
        if (action === "send") return this.action(`/annotation-runs/${this.annotationId}/cvat`, "已提交图片与标签，请等待审核工作区准备完成");
        if (action === "seal") return this.action(`/annotation-runs/${this.annotationId}/exports`, "已提交数据封存，完成后请前往训练页选择数据");
        if (action === "review") {
          if (this.page !== "review") { this.preparationFocus = "cvat-review"; return this.navigate("review"); }
          return this.read(this.openCvatReview());
        }
        if (action === "tasks") {
          this.preparationFocus = "annotation-tasks";
          if (this.page !== "annotate") return this.navigate("annotate");
          this.focusPreparationSection();
        }
      },
      focusPreparationSection() {
        if (!this.preparationFocus) return;
        const id = this.preparationFocus;
        this.preparationFocus = "";
        this.$nextTick(() => {
          const target = document.getElementById(id) || (id === "cvat-review" ? document.getElementById("cvat-connection") : null);
          target?.scrollIntoView({ block: "start", behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        });
      },
      sourceName(source) { return ({raw_images:"原图自动标注",labeled_upload:"上传标签 · 待审核",reviewed_upload:"上传已审核数据"})[source] || "原图"; },
      dataState(state) { return ({ready:"可直接训练",reviewed:"审核完成 · 待准备训练数据",needs_review:"等待人工审核",unlabeled:"等待标注",annotating:"正在标注",stopped:"已停止"})[state] || "等待上传"; },
      async changeDataScope() {
        this.annotationId = ""; this.annotationPinned = true;
        this.annotationTasks = []; this.annotationImages = []; this.preparationFocus = "";
        this.annotationImageOffset = 0; this.annotationImageKey = this.annotationQualityKey = "";
        this.cvatSessionUrl = this.cvatSessionKey = this.cvatAttemptKey = this.cvatError = "";
        await this.chooseDataset(this.datasetId);
        if (this.scopeVersion?.annotation_run_id) await this.chooseAnnotation(this.scopeVersion.annotation_run_id);
      },
      async changeVersionScope() {
        this.annotationTasks = []; this.annotationImages = []; this.preparationFocus = "";
        this.annotationImageOffset = 0; this.annotationImageKey = this.annotationQualityKey = "";
        this.cvatSessionUrl = this.cvatSessionKey = this.cvatAttemptKey = this.cvatError = "";
        this.annotationId = this.scopeVersion?.annotation_run_id || "";
        this.annotationQuality = null; this.annotationTasks = [];
        if (this.annotationId) await this.chooseAnnotation(this.annotationId);
      },
      useReadyDataset(exportId) {
        this.newTraining(); this.trainForm.exportId = exportId; this.navigate('train');
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
        const qualityKey = `${id}:${run.status}:${run.processed}`;
        const imageKey = `${run.version_id}:${this.annotationImageOffset}`;
        const [quality, images, tasks] = await Promise.all([
          this.annotationQualityKey === qualityKey ? Promise.resolve(this.annotationQuality) : api(`/annotation-runs/${id}/quality`),
          this.annotationImageKey === imageKey ? Promise.resolve(this.annotationImages) : api(`/dataset-versions/${run.version_id}/images?limit=24&offset=${this.annotationImageOffset}`),
          api(`/annotation-runs/${id}/tasks?limit=20`),
        ]);
        if (id !== this.annotationId || ticket !== detailTickets.annotation)
          return;
        this.annotationQuality = quality;
        this.annotationQualityKey = qualityKey;
        this.annotationImageKey = imageKey;
        this.annotationImages = images;
        this.annotationTasks = tasks;
        const index = this.annotations.findIndex((x) => x.id === id);
        if (index >= 0) this.annotations[index] = run;
        if (this.page === "review" && run.cvat_task_id) await this.connectCvat();
      },
      /** 同源请求建立 Cookie 会话后再加载嵌入编辑器；轮询不重复登录或刷新正在编辑的作业。 */
      async connectCvat(force = false) {
        const run = this.currentAnnotation;
        if (!run?.cvat_task_id) return;
        const key = `${run.id}:${run.cvat_task_id}`;
        if (!force && this.cvatAttemptKey === key) return;
        const ticket = ++detailTickets.cvat;
        this.cvatAttemptKey = key;
        this.cvatConnecting = true;
        this.cvatError = "";
        this.cvatSessionUrl = this.cvatSessionKey = "";
        try {
          const result = await api(`/annotation-runs/${run.id}/cvat-session`, { method: "POST" });
          if (ticket !== detailTickets.cvat || this.annotationId !== run.id || this.page !== "review") return;
          const target = new URL(result.url);
          if (!['http:', 'https:'].includes(target.protocol) || target.hostname !== location.hostname || target.username || target.password)
            throw new Error("审核服务地址与当前工作台不匹配");
          this.cvatSessionKey = key;
          this.cvatSessionUrl = target.href;
        } catch (error) {
          if (ticket === detailTickets.cvat) this.cvatError = error.message;
        } finally {
          if (ticket === detailTickets.cvat) this.cvatConnecting = false;
        }
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
          this.openImage(`/api/v1/images/${preview.image.id}/view`, preview.image.original_name,
            preview.boxes.filter(box => box.review_status !== "rejected").map(box => ({
              bbox: [box.bbox[0] / preview.image.width, box.bbox[1] / preview.image.height, (box.bbox[2] - box.bbox[0]) / preview.image.width, (box.bbox[3] - box.bbox[1]) / preview.image.height],
              label: `${box.class_name} · ${box.confidence == null ? this.status(box.review_status) : this.metric(box.confidence, 2)}`, status: "tp",
            })), preview.boxes);
        } catch (error) {
          this.error = error.message;
        }
      },
      openImage(src, title, boxes = [], details = null) {
        this.viewer = { src, title, boxes, details, zoom: 1, fitWidth: 800, naturalWidth: 0, showBoxes: true, loading: true, error: "" };
        this.$nextTick(() => { if (!this.$refs.imageViewer.open) this.$refs.imageViewer.showModal(); });
      },
      imageLoaded(event) {
        if (!this.viewer) return;
        const image = event.target, viewport = this.$refs.viewerViewport;
        this.viewer.naturalWidth = image.naturalWidth;
        this.viewer.fitWidth = Math.min(image.naturalWidth, Math.max(240, viewport.clientWidth - 48), Math.max(200, viewport.clientHeight - 48) * image.naturalWidth / image.naturalHeight);
        this.viewer.loading = false;
      },
      closeImage() { this.$refs.imageViewer.close(); this.viewer = null; },
      zoomImage(factor) { if (this.viewer) this.viewer.zoom = Math.max(0.25, Math.min(16, this.viewer.zoom * factor)); },
      openErrorImage(item) {
        const classes = this.evaluationReport?.per_class || [];
        const boxes = [...item.ground_truth.map(box => ({ ...box, truth: true })), ...item.predictions].map(box => ({ ...box, label: `${box.truth ? "真值" : "预测"} · ${box.label || classes.find(row => row.class_id === box.class_id)?.name || "目标"} · ${box.status.toUpperCase()}` }));
        const { imageStatus, ...details } = item;
        this.openImage(`/api/v1/evaluations/${this.evaluationId}/images/${item.image_id}`, `错误定位 · TP ${item.counts.tp} / FP ${item.counts.fp} / FN ${item.counts.fn}`, boxes, details);
      },
      viewerBoxStyle(box) {
        return { left: `${box.bbox[0] * 100}%`, top: `${box.bbox[1] * 100}%`, width: `${box.bbox[2] * 100}%`, height: `${box.bbox[3] * 100}%` };
      },
      /** 只进入当前项目关联的审核批次，编辑由嵌入的官方 CVAT 完成。 */
      async openCvatReview(runId = this.annotationId) {
        if (runId && runId !== this.annotationId) {
          if (!this.annotations.some(run => run.id === runId)) return;
          await this.chooseAnnotation(runId);
        }
        if (!this.cvatEmbedUrl) await this.connectCvat();
        this.preparationFocus = this.cvatEmbedUrl ? "cvat-review" : "cvat-connection";
        this.focusPreparationSection();
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
          this.resetTrainingLogFollow();
          this.experiment = null;
          this.trainingTasks = [];
          this.trainingTaskId = "";
          this.trainingLogs = "";
          this.metrics = [];
          this.epochs = [];
          this.trainingCharts = [];
          this.evaluationId = "";
          this.evaluationReport = null;
          this.errorImages = [];
          this.evaluations = [];
        }
        this.experimentId = id;
        const [experiment, tasks, evaluations, chartArtifacts] = await Promise.all([
          api(`/experiments/${id}`),
          api(`/experiments/${id}/tasks`),
          api(`/evaluations?experiment_id=${id}&limit=50`),
          api(`/experiments/${id}/charts`),
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
        this.trainingCharts = chartArtifacts;
        this.metrics = rows;
        this.epochs = epochRows(rows);
        this.evaluations = evaluations;
        if (this.trainingTaskId !== taskId) this.resetTrainingLogFollow();
        this.trainingTaskId = taskId;
        this.trainingLogs = logs;
        const logView = trainingLogView;
        this.$nextTick(() => {
          if (logView === trainingLogView) this.scrollTrainingLog();
        });
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
      /** 日志自动跟随仅响应真实输入意图，程序滚动和内容更新不重置空闲计时。 */
      scrollTrainingLog() {
        if (!this.trainingLogAutoFollow || this.page !== "train" || document.hidden) return;
        const log = this.$refs.trainingLog;
        if (log?.clientHeight) log.scrollTop = log.scrollHeight;
      },
      resetTrainingLogFollow() {
        ++trainingLogView;
        clearTimeout(trainingLogResumeTimer);
        trainingLogResumeTimer = null;
        trainingLogPointer = null;
        trainingLogTouching = false;
        this.trainingLogAutoFollow = true;
      },
      pauseTrainingLogFollow() {
        clearTimeout(trainingLogResumeTimer);
        trainingLogResumeTimer = null;
        this.trainingLogAutoFollow = false;
        if (trainingLogPointer !== null || trainingLogTouching) return;
        const view = trainingLogView;
        trainingLogResumeTimer = setTimeout(() => {
          trainingLogResumeTimer = null;
          if (view !== trainingLogView) return;
          this.trainingLogAutoFollow = true;
          this.$nextTick(() => {
            if (view === trainingLogView) this.scrollTrainingLog();
          });
        }, 10_000);
      },
      startTrainingLogInteraction(event) {
        trainingLogPointer = event.pointerId;
        this.pauseTrainingLogFollow();
      },
      endTrainingLogInteraction(event) {
        if (trainingLogPointer === null || (event && event.pointerId !== trainingLogPointer)) return;
        trainingLogPointer = null;
        this.pauseTrainingLogFollow();
      },
      startTrainingLogTouch() {
        trainingLogTouching = true;
        this.pauseTrainingLogFollow();
      },
      endTrainingLogTouch(event) {
        if (!trainingLogTouching || event.touches.length) return;
        trainingLogTouching = false;
        this.pauseTrainingLogFollow();
      },
      trainingLogKeydown(event) {
        if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) this.pauseTrainingLogFollow();
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
        const imageStates = new Map(this.errorImages.map(item => [item.image_id, item.imageStatus]));
        this.errorImages = errors.items.map(item => ({ ...item, imageStatus: imageStates.get(item.image_id) || "loading" }));
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
          this.resetAgentChat();
          this.agent = null;
          this.agentBudget = null;
          this.agentIterations = [];
          this.annotationPinned = this.experimentPinned = false;
          if (!this.trainNew) {
            this.trainFormKey = "";
            if (this.page === "train") this.resetTrainForm("new");
          }
        }
        if (reset || id !== this.agentId) {
          this.eventCursor = 0;
          this.agentEvents = [];
        }
        this.agentId = id;
        const [run, iterations, budget, events, messages, instructions] = await Promise.all([
          api(`/agent-runs/${id}`),
          api(`/agent-runs/${id}/iterations`),
          api(`/agent-runs/${id}/budget`),
          api(`/agent-runs/${id}/events?after_id=${this.eventCursor}&limit=500`),
          api(`/agent-runs/${id}/operator-messages`),
          api(`/agent-runs/${id}/operator-instructions`),
        ]);
        if (id !== this.agentId || ticket !== detailTickets.agent) return;
        const thread = this.$refs.agentChatThread;
        const follow = !thread || thread.scrollHeight - thread.scrollTop - thread.clientHeight < 64;
        const messageKey = rows => rows.map(row => `${row.request_id}:${row.answered_at || ""}`).join("|");
        const messagesChanged = messageKey(messages) !== messageKey(this.operatorMessages);
        this.agent = run;
        this.operatorMessages = messages;
        this.operatorInstructions = instructions;
        this.agentIterations = iterations;
        this.agentBudget = budget;
        const ids = new Set(this.agentEvents.map((x) => x.id));
        this.agentEvents.push(...events.filter((x) => !ids.has(x.id)));
        if (events.length) this.eventCursor = events.at(-1).id;
        if (this.agentChatOpen && follow && messagesChanged) this.$nextTick(() => this.scrollAgentChat());
        if (this.page === "train") this.syncTrainingForm();
      },
      resetAgentChat() {
        ++agentChatTicket;
        this.operatorMessages = [];
        this.operatorInstructions = [];
        this.agentChatOpen = false;
        this.agentQuestion = this.agentWakeError = this.agentWakeAcceptedUntil = "";
        this.agentWakeSending = false;
      },
      openAgentChat() {
        this.agentChatOpen = true;
        this.$nextTick(() => {
          this.scrollAgentChat();
          if (this.canWakeAgent) this.$refs.agentQuestion?.focus();
          else this.$refs.agentChatHeading?.focus();
        });
      },
      async revokeOperatorInstruction(item) {
        if (this.busy || !this.agentId) return;
        const id = this.agentId, project = this.projectId;
        this.busy = true;
        this.agentWakeError = "";
        try {
          await api(`/agent-runs/${id}/operator-instructions/${encodeURIComponent(item.key)}/revoke`, {
            method: "POST", body: { instruction_id: item.instruction_id },
          });
          if (id !== this.agentId || project !== this.projectId) return;
          const instructions = await api(`/agent-runs/${id}/operator-instructions`);
          if (id !== this.agentId || project !== this.projectId) return;
          this.operatorInstructions = instructions;
          this.toast("持续指令已撤销，将从下一次决策起生效");
        } catch (error) {
          if (id === this.agentId && project === this.projectId) this.agentWakeError = error.message;
        } finally { this.busy = false; }
      },
      scrollAgentChat() {
        const thread = this.$refs.agentChatThread;
        thread?.scrollTo({ top: thread.scrollHeight, behavior: "auto" });
      },
      closeAgentChat() {
        this.agentChatOpen = false;
        this.$nextTick(() => this.$refs.agentChatToggle?.focus());
      },
      async wakeAgent(event) {
        if (event?.isComposing || this.busy || this.agentWakeSending || !this.canWakeAgent) return;
        const id = this.agentId, project = this.projectId;
        const ticket = agentChatTicket;
        const question = this.agentQuestion.trim();
        const until = this.agent.state.waiting.until;
        this.agentWakeSending = true;
        this.agentWakeError = "";
        try {
          await api(`/agent-runs/${id}/wake`, { method: "POST", body: { question, sleep_until: until } });
          if (ticket !== agentChatTicket || id !== this.agentId || project !== this.projectId) return;
          this.agentWakeAcceptedUntil = until;
          if (this.agentQuestion.trim() === question) this.agentQuestion = "";
          this.toast(question ? "已请求唤醒，问题已交给 Agent" : "已请求立即唤醒 Agent");
          try { await this.chooseAgent(id, false); }
          catch (_) { this.agentWakeError = "唤醒请求已提交，暂未刷新到最新状态，请稍后刷新。"; }
        } catch (error) {
          if (ticket === agentChatTicket && id === this.agentId && project === this.projectId) this.agentWakeError = error.message;
        } finally {
          if (ticket === agentChatTicket && id === this.agentId && project === this.projectId) this.agentWakeSending = false;
        }
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
        return { yolo26_coco: "COCO 预训练", project_model: "项目模型", grounding_dino: "DINO", human_added_class: "人工新增类别", external_reviewed_yolo: "已审标签导入", external_labeled_yolo: "上传已有标签" }[source] || source;
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
        if (!files.length || this.busy || this.datasetUploading) return;
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
          const transfer = this.beginTransfer("import", files, body);
          input.value = "";
          this.uploadCount = 0;
          await transfer;
        } catch (error) {
          this.inlineError = error.message;
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
        if (this.busy || this.datasetUploading || !this.datasetId || !labeledFiles.length) return;
        const options = this.labeledUpload;
        this.inlineError = '';
        if (options.reviewed && (!options.reviewer.trim() || !options.confirmation.trim())) {
          this.inlineError = '请填写外部审核者和审核来源说明，或选择待审核导入';
          return;
        }
        try {
          const body = new FormData();
          body.append('options', JSON.stringify({ accepted_reviewed_source: options.reviewed, reviewer: options.reviewer, confirmation: options.confirmation }));
          for (const file of labeledFiles) body.append('files', file, file.webkitRelativePath || file.name);
          const transfer = this.beginTransfer("labeled_import", labeledFiles, body);
          labeledFiles = [];
          options.name = ''; options.count = 0; options.bytes = 0;
          for (const ref of ['labeledFolder', 'labeledArchive']) if (this.$refs[ref]) this.$refs[ref].value = '';
          await transfer;
        } catch (error) {
          this.inlineError = error.message;
        }
      },
      /** 上传属于应用级任务；路由切换不会释放 XHR，也不会改动已捕获的数据集归属。 */
      async beginTransfer(kind, files, body) {
        // 仅用于本页列表身份；业务 UUID 由服务端生成，普通 HTTP 站点也可上传。
        const key = `upload-${Date.now()}-${++uploadSequence}`;
        this.uploads.push({ key, id: "", kind, datasetId: this.datasetId, projectId: this.projectId,
          datasetName: this.datasets.find(row => row.id === this.datasetId)?.name || "数据集",
          projectName: this.currentProject?.name || "项目", count: files.length, phase: "preparing",
          loaded: 0, total: files.reduce((sum, file) => sum + file.size, 0), percent: 0, processed: 0, error: "" });
        const transfer = this.uploads.find(item => item.key === key);
        try {
          const task = await api(`/datasets/${transfer.datasetId}/uploads`, { method: "POST", body: { kind, total: files.length } });
          transfer.id = task.id;
          transfer.phase = "uploading";
          const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            uploadRequests.set(key, xhr);
            xhr.open("POST", `${API}/datasets/${transfer.datasetId}/${kind === "import" ? "imports" : "labeled-imports"}?upload_id=${encodeURIComponent(task.id)}`);
            xhr.upload.onprogress = event => {
              if (transfer.phase === "canceling") return;
              transfer.loaded = event.loaded;
              if (event.lengthComputable) { transfer.total = event.total; transfer.percent = Math.round(event.loaded / event.total * 100); }
            };
            xhr.upload.onload = () => { if (transfer.phase !== "canceling") { transfer.percent = 100; transfer.phase = "receiving"; } };
            xhr.onload = () => {
              let result;
              try { result = JSON.parse(xhr.responseText); } catch (_) { result = null; }
              if (xhr.status >= 200 && xhr.status < 300) resolve(result);
              else reject(new Error(typeof result?.detail === "string" ? result.detail : `上传失败（HTTP ${xhr.status}）`));
            };
            xhr.onabort = () => reject(new DOMException("上传已取消", "AbortError"));
            xhr.onerror = () => reject(new Error("上传连接中断，请查看后台任务确认接收状态"));
            xhr.send(body);
          });
          transfer.phase = result.status;
          transfer.processed = result.processed;
          transfer.taskTotal = result.total;
          transfer.percent = 100;
          this.toast(`${transfer.datasetName}：文件已接收，可继续浏览其他页面`);
        } catch (error) {
          if (transfer.phase !== "stopped") {
            transfer.error = error.message;
            // 响应丢失也可能已入队，先按持久状态核对，不能虚报取消成功。
            if (transfer.id) {
              try {
                const task = await api(`/dataset-tasks/${transfer.id}`);
                if (task.status === "uploading") await api(`/dataset-tasks/${transfer.id}/cancel-upload`, { method: "POST" });
                transfer.phase = task.status === "uploading" ? "failed" : task.status;
              } catch (_) { transfer.phase = "unknown"; }
            } else transfer.phase = "failed";
          }
        } finally {
          uploadRequests.delete(key);
          await this.refresh();
        }
      },
      async cancelTransfer(transfer) {
        const previous = transfer.phase;
        transfer.phase = "canceling";
        try {
          await api(`/dataset-tasks/${transfer.id}/cancel-upload`, { method: "POST" });
          transfer.phase = "stopped";
          transfer.error = "";
          uploadRequests.get(transfer.key)?.abort();
        } catch (error) {
          transfer.phase = previous;
          transfer.error = error.message;
          try {
            const task = await api(`/dataset-tasks/${transfer.id}`);
            if (task.status !== "uploading") { transfer.phase = task.status; transfer.processed = task.processed; transfer.taskTotal = task.total; }
          } catch (_) { /* 保留取消失败提示，传输仍由原 XHR 管理。 */ }
        }
      },
      async cancelDatasetUpload(id) {
        const transfer = this.uploads.find(item => item.id === id);
        if (transfer) return this.cancelTransfer(transfer);
        try {
          await api(`/dataset-tasks/${id}/cancel-upload`, { method: "POST" });
          this.toast("已取消未完成的上传，可重新选择文件");
          await this.refresh();
        } catch (error) { this.inlineError = error.message; }
      },
      async pollUploads() {
        await Promise.all(this.uploads.filter(item => item.id && ["queued", "running", "unknown"].includes(item.phase)).map(async item => {
          try {
            const task = await api(`/dataset-tasks/${item.id}`);
            item.phase = task.status; item.processed = task.processed; item.taskTotal = task.total;
            if (task.status === 'completed' && task.result?.version_id && item.datasetId === this.datasetId) this.scopeVersionId = task.result.version_id;
            item.error = task.error || "";
          } catch (error) { item.error = error.message; }
        }));
      },
      transferText(item) {
        return ({ preparing: "正在登记上传", uploading: `正在传输 · ${item.percent}%`, receiving: "传输 100% · 服务器正在接收", canceling: "正在取消上传", queued: "文件已接收 · 等待后台检查", running: `后台检查 · ${item.processed} / ${item.taskTotal ?? item.count}`, completed: "导入完成", stopped: "上传已取消", failed: "上传失败", unknown: "接收状态待确认" })[item.phase] || this.status(item.phase);
      },
      get datasetUploading() {
        return this.uploads.some(item => item.datasetId === this.datasetId && ["preparing", "uploading", "receiving", "canceling", "queued", "running", "unknown"].includes(item.phase)) || this.datasetTasks.some(item => item.dataset_id === this.datasetId && ["uploading", "queued", "running", "canceling"].includes(item.status));
      },
      datasetAnnotation(versionId) {
        return this.annotations.filter(run => run.version_id === versionId && ['pending_review','reviewed','exported'].includes(run.status))
          .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id))[0];
      },
      openDatasetReview(versionId) {
        const run = this.datasetAnnotation(versionId);
        if (!run || this.busy) return;
        this.annotationId = run.id;
        this.scopeVersionId = versionId;
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
        this.resetPromptOptimization();
        const policy = this.overview?.configuration?.agent_defaults || {};
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
          exportId: "",
          requirement: this.agent?.requirement || "",
          name: "",
          lockModel: policy.allowed_models?.length === 1,
        };
        if (this.trainForm.lockModel) this.trainForm.params.model = policy.allowed_models[0];
        this.trainFormKey = key;
      },
      syncTrainingForm() {
        if (this.showTrainForm) {
          const key = "new";
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
        this.resetPromptOptimization();
        const parameters = iteration?.plan?.parameters || Object.fromEntries(
          (this.agent.state?.operator_fields || []).map(field => [field, this.agent.initial_parameters?.[field]]),
        );
        const params = { ...emptyParams(), ...Object.fromEntries(this.trainingParameterGroups.flatMap(group => group.fields.map(field => [field.name, ""]))) };
        for (const field of Object.keys(params)) {
          const value = parameters[field];
          if (value == null) continue;
          params[field] = typeof value === "boolean" ? String(value) : Array.isArray(value) ? JSON.stringify(value) : value;
        }
        if (params.tile === "" && parameters.tile == null) params.tile = "false";
        this.trainForm = {
          params,
          budget: { ...this.agent.policy },
          exportId: iteration?.plan?.dataset_export_id || this.agent.dataset_export_id || "",
          requirement: this.agent.requirement || "",
          name: this.agent.name,
          lockModel: this.agent.policy.allowed_models?.length === 1,
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
      /** 优化建议与训练创建共享同一份表单约束，未填写的参数不补默认值。 */
      trainingPolicy(params) {
        const budget = this.trainForm.budget;
        const policy = {
          max_experiments: Number(budget.max_experiments),
          max_training_hours: Number(budget.max_training_hours),
          max_epochs_per_experiment: Number(budget.max_epochs_per_experiment),
          max_imgsz: Number(budget.max_imgsz),
          stagnation_rounds: Number(budget.stagnation_rounds),
          minimum_improvement: Number(budget.minimum_improvement),
          purpose: budget.purpose,
          allowed_models: this.trainForm.lockModel ? [params.model] : (this.overview?.configuration?.agent_defaults?.allowed_models || ["yolo26n.pt", "yolo26s.pt"]),
        };
        if (this.trainForm.lockModel && !params.model) throw new Error("锁定模型前，请先选择一个基座模型");
        if (params.model && !policy.allowed_models.includes(params.model)) throw new Error("所选模型不在本次允许范围内，请锁定该模型或选择允许的模型");
        return policy;
      },
      trainingPromptInput() {
        const params = this.pinnedParams();
        return {
          project_id: this.projectId,
          requirement: this.trainForm.requirement.trim(),
          dataset_export_id: this.trainForm.exportId || null,
          initial_parameters: params,
          policy: this.trainingPolicy(params),
        };
      },
      resetPromptOptimization() {
        ++promptOptimizationTicket;
        promptOptimizationRequest?.abort();
        promptOptimizationRequest = null;
        this.promptOptimizer = { busy: false, error: "", draft: "", notes: "", context: "", original: "", appliedText: "" };
      },
      /** 仅在用户点击时请求建议；迟到响应不能跨项目或覆盖新表单。 */
      async optimizeTrainingPrompt() {
        if (this.busy || this.promptOptimizer.busy || !this.showTrainForm || !this.llmReady) return;
        let body;
        try {
          body = this.trainingPromptInput();
          if (body.requirement.length < 4) throw new Error("请先填写至少 4 个字的训练需求");
        } catch (error) {
          this.promptOptimizer.error = error.message;
          return;
        }
        this.resetPromptOptimization();
        const ticket = promptOptimizationTicket;
        promptOptimizationRequest = new AbortController();
        this.promptOptimizer.busy = true;
        this.promptOptimizer.original = this.trainForm.requirement;
        this.promptOptimizer.context = JSON.stringify(body);
        try {
          const draft = await api("/training-prompts/optimize", { method: "POST", body, signal: promptOptimizationRequest.signal });
          if (ticket !== promptOptimizationTicket) return;
          this.promptOptimizer.draft = draft.requirement;
          this.promptOptimizer.notes = draft.notes || "";
        } catch (error) {
          if (ticket === promptOptimizationTicket && error.name !== "AbortError") this.promptOptimizer.error = error.message;
        } finally {
          if (ticket === promptOptimizationTicket) {
            this.promptOptimizer.busy = false;
            promptOptimizationRequest = null;
          }
        }
      },
      get promptOptimizationStale() {
        if (!this.promptOptimizer.context) return false;
        try { return this.promptOptimizer.context !== JSON.stringify(this.trainingPromptInput()); }
        catch (_) { return true; }
      },
      applyTrainingPrompt() {
        if (this.busy || !this.showTrainForm || this.promptOptimizationStale) return;
        const text = this.promptOptimizer.draft.trim();
        if (text.length < 4 || text.length > 2000) {
          this.promptOptimizer.error = "优化后的需求需为 4–2000 个字符";
          return;
        }
        this.trainForm.requirement = text;
        this.promptOptimizer.appliedText = text;
        this.promptOptimizer.error = "";
        this.toast("已采用优化建议，仍可继续编辑需求");
        this.$nextTick(() => this.$refs.trainingRequirement?.focus());
      },
      restoreTrainingPrompt() {
        if (this.busy || !this.showTrainForm || this.trainForm.requirement !== this.promptOptimizer.appliedText) return;
        const original = this.promptOptimizer.original;
        this.resetPromptOptimization();
        this.trainForm.requirement = original;
        this.toast("已恢复优化前的原文");
        this.$nextTick(() => this.$refs.trainingRequirement?.focus());
      },
      /** 明确选择已审数据后创建独立训练流程。
       * @returns {Promise<void>} 请求结束。
       */
      async startTraining() {
        if (this.busy || !this.showTrainForm) return;
        this.busy = true;
        this.inlineError = "";
        try {
          const params = this.pinnedParams();
          const policy = this.trainingPolicy(params);
          if (params.epochs > policy.max_epochs_per_experiment)
            throw new Error("首轮 epochs 超过单实验上限");
          if (params.imgsz > policy.max_imgsz)
            throw new Error("首轮输入尺寸超过上限");
          if (!this.trainForm.exportId) throw new Error("请选择一份已完成标注和审核的训练数据");
          this.resetPromptOptimization();
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
      async openDatasetCopy() {
        if (this.busy) return;
        const project = this.projectId;
        this.busy = true;
        this.inlineError = "";
        try {
          const rows = await api('/workspace/datasets');
          if (project !== this.projectId) return;
          this.copySources = rows.filter(row => row.project_id !== project && row.versions.some(version => ['ready','reviewed'].includes(version.status)));
          ++this.copySourceTicket;
          this.copyVersions = [];
          this.copyVersionsLoading = false;
          await this.show("copyDataset");
          this.form = { source_id: "", version_id: "", name: "" };
        } catch (error) { this.inlineError = error.message; }
        finally { this.busy = false; }
      },
      async chooseCopySource() {
        const id = this.form.source_id;
        const ticket = ++this.copySourceTicket;
        const source = this.copySources.find(row => row.id === id);
        this.copyVersions = [];
        this.form.version_id = "";
        this.form.name = source ? `${source.name} · 副本`.slice(0, 200) : "";
        this.formError = "";
        this.copyVersionsLoading = true;
        try {
          const rows = source?.versions || [];
          if (ticket !== this.copySourceTicket || this.modal !== "copyDataset") return;
          this.copyVersions = rows.filter(row => ['ready','reviewed'].includes(row.status));
          this.form.version_id = this.copyVersions[0]?.id || "";
        } catch (error) { if (ticket === this.copySourceTicket) this.formError = error.message; }
        finally { if (ticket === this.copySourceTicket) this.copyVersionsLoading = false; }
      },
      copySourceLabel(source) {
        const project = this.projects.find(row => row.id === source.project_id);
        return `${project?.name || "其他项目"} / ${source.name}`;
      },
      /** 发布或异步提交中不允许误关表单导致操作状态丢失。
       * @returns {void} 无。
       */
      deleteDataset() {
        if (!this.currentDataset || this.datasetUploading || this.busy) return;
        const dataset = this.currentDataset;
        this.show("deleteAsset");
        this.form = { kind: "dataset", id: dataset.id, name: dataset.name, confirm: "" };
      },
      deleteImage(image) {
        if (this.datasetUploading || this.busy) return;
        this.show("deleteAsset");
        this.form = { kind: "image", id: image.id, name: image.original_name, confirm: "" };
      },
      async confirmDelete() {
        if (this.busy || this.form.confirm !== this.form.name) return;
        this.busy = true;
        this.formError = "";
        try {
          const { kind, id, name } = this.form;
          await api(`/${kind === "dataset" ? "datasets" : "images"}/${id}`, { method: "DELETE", body: { confirm_name: name } });
          if (kind === "dataset") {
            const versions = new Set(this.versions.filter(row => row.dataset_id === id).map(row => row.id));
            this.datasets = this.datasets.filter(row => row.id !== id);
            this.versions = this.versions.filter(row => !versions.has(row.id));
            this.annotations = this.annotations.filter(row => !versions.has(row.version_id));
            this.exports = this.exports.filter(row => !versions.has(row.version_id));
            this.datasetId = this.datasets[0]?.id || "";
            this.images = []; this.datasetVersions = []; this.datasetTasks = []; this.imageOffset = 0;
          } else this.images = this.images.filter(row => row.id !== id);
          this.busy = false;
          this.closeDialog();
          this.toast("已从工作区删除，封存版本与历史记录保持可追溯");
          await this.refresh();
        } catch (error) { this.formError = error.message; }
        finally { this.busy = false; }
      },
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
        if (this.modal === "deleteAsset") return this.confirmDelete();
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
          if (this.modal === "copyDataset") {
            if (!f.source_id || !f.name.trim() || this.copyVersionsLoading) throw new Error("请选择来源数据集并填写副本名称");
            result = await api(`/datasets/${f.source_id}/copies`, {
              method: "POST", body: { project_id: this.projectId, name: f.name.trim(), version_id: f.version_id || null },
            });
            this.datasetId = result.dataset_id;
            message = "副本已创建，后台正在复制图片和所选版本；可在后台任务查看进度";
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
            message = "已提交审核回传，正在核验 CVAT 作业；成功后请封存为训练数据";
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
                ? "已提交审核回传，核验成功后自动封存；完成后请到训练页手动开始训练"
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
        if (key === "tile") {
          if (value === true || value === "true") return true;
          if (value === false || value === "false") return false;
          const size = Number(value);
          if (!Number.isInteger(size) || size <= 0) throw new Error("tile 请填 true、false 或正整数窗口边长");
          return size;
        }
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
        if (this.page === "train") for (const item of this.trainingMetricCharts) {
          const canvas = document.getElementById(`metric-${item.key}`);
          if (!canvas) continue;
          const key = item.aliases.find(name => this.epochs.some(row => row[name] != null));
          if (!key) { charts.get(`metric-${item.key}`)?.destroy(); charts.delete(`metric-${item.key}`); continue; }
          this.chart(`metric-${item.key}`, canvas, { labels, datasets: [series(key, item.title, item.score ? colors[0] : colors[1])] }, options(false, item.score ? { min: 0, max: 1 } : {}));
        }
      },
      get trainingMetricCharts() {
        return [
          { key: "train-box", title: "训练 · 框 Loss", aliases: ["train/box_loss"] },
          { key: "train-cls", title: "训练 · 分类 Loss", aliases: ["train/cls_loss"] },
          { key: "train-l1", title: "训练 · L1 / DFL Loss", aliases: ["train/l1_loss", "train/dfl_loss"] },
          { key: "precision", title: "验证 · Precision 精确率", aliases: ["metrics/precision(B)"], score: true },
          { key: "recall", title: "验证 · Recall 召回率", aliases: ["metrics/recall(B)"], score: true },
          { key: "val-box", title: "验证 · 框 Loss", aliases: ["val/box_loss"] },
          { key: "val-cls", title: "验证 · 分类 Loss", aliases: ["val/cls_loss"] },
          { key: "val-l1", title: "验证 · L1 / DFL Loss", aliases: ["val/l1_loss", "val/dfl_loss"] },
          { key: "map50", title: "验证 · mAP50", aliases: ["metrics/mAP50(B)"], score: true },
          { key: "map95", title: "验证 · mAP50–95", aliases: ["metrics/mAP50-95(B)"], score: true },
        ].map(item => ({ ...item, available: this.epochs.some(row => item.aliases.some(key => row[key] != null)) }));
      },
      chartTitle(name) { return chartNames[name] || name; },
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
      /** 名称用于理解，UUID 仅用于追溯；没有加载的历史实体不猜测身份。 */
      entityLabel(kind, id) {
        if (!id) return "尚未生成";
        if (kind === "experiment") {
          const row = this.experiments.find(item => item.id === id);
          const iteration = this.agentIterations.find(item => item.experiment_id === id);
          const parameters = row?.parameters || iteration?.plan?.parameters;
          if (iteration) return `第 ${iteration.iteration + 1} 轮${parameters?.model ? ` · ${parameters.model.replace('.pt', '').toUpperCase()} / ${parameters.imgsz}px` : ""}`;
          return row?.name || "历史实验（详情未加载）";
        }
        if (kind === "model") {
          const row = this.models.find(item => item.id === id);
          const best = this.agent?.state?.best_model;
          const experimentId = row?.experiment_id || (best?.model_id === id ? best.experiment_id : null);
          return experimentId ? `${this.entityLabel("experiment", experimentId)} · ${row?.kind === "last" ? "最后一次保存权重" : "最佳权重 best.pt"}` : "历史模型权重（详情未加载）";
        }
        if (kind === "task") {
          const index = this.trainingTasks.findIndex(item => item.id === id);
          return index >= 0 ? `训练尝试 ${this.trainingTasks.length - index}` : "历史训练尝试";
        }
        if (kind === "export") {
          const row = this.exports.find(item => item.id === id);
          const version = this.versions.find(item => item.id === row?.version_id);
          const dataset = this.datasets.find(item => item.id === version?.dataset_id);
          return row ? `${dataset?.name || "已审训练数据"}${version ? ` · v${version.number}` : ""} · ${row.summary?.image_count ?? "—"} 张` : "已审训练数据（详情未加载）";
        }
        if (kind === "version") {
          const row = this.versions.find(item => item.id === id);
          return row ? `v${row.number} · ${row.name}` : "历史数据版本";
        }
        if (kind === "annotation") {
          const row = this.annotations.find(item => item.id === id);
          return row ? `${row.total} 张 · ${this.time(row.created_at)}` : "历史标注批次";
        }
        if (kind === "evaluation") {
          const row = this.evaluations.find(item => item.id === id);
          return row ? `验证评估 · ${this.time(row.created_at)}` : "历史验证评估";
        }
        return "历史记录";
      },
      readableText(value) {
        let result = String(value || "");
        for (const [kind, rows] of [["experiment", this.experiments], ["model", this.models], ["task", this.trainingTasks], ["export", this.exports], ["annotation", this.annotations], ["version", this.versions]]) {
          for (const row of rows) {
            result = result.replaceAll(row.id, this.entityLabel(kind, row.id));
            result = result.replace(new RegExp(`\\b${row.id.slice(0, 8)}\\b`, "g"), this.entityLabel(kind, row.id));
          }
        }
        return result.replace(/\b[a-f0-9]{8}-[a-f0-9-]{27,}\b/gi, "关联记录（编号见原文）").replace(/\b[a-f0-9]{8}\b/gi, "历史记录（编号见原文）");
      },
      textPreview(value, limit = 110) {
        const text = this.readableText(value).replace(/\s+/g, " ").trim();
        if (text.length <= limit) return text;
        const sentence = text.slice(0, limit).match(/^.*?[。！？；]/)?.[0];
        return sentence || text.slice(0, limit) + "…";
      },
      textBlocks(value) {
        return this.readableText(value).split(/\n\s*\n|\n|(?<=[。；])\s*(?=\S)/).map(text => text.trim()).filter(Boolean);
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
        if (['upload','annotate','review'].includes(this.page)) {
          const version = this.scopeVersion;
          if (step === 1) return this.scopeDataset?.image_count ? 'done' : 'current';
          if (step === 2) return version?.source && version.source !== 'raw_images' ? 'skipped' : version?.annotation_run_id && ['needs_review','reviewed','ready'].includes(version.status) ? 'done' : this.page === 'annotate' ? 'current' : 'todo';
          if (step === 3) return ['reviewed','ready'].includes(version?.status) ? 'done' : version?.status === 'needs_review' ? 'gate' : version?.status === 'stopped' ? 'stopped' : 'todo';
          return 'todo';
        }
        const run = this.agent;
        if (!run) return step === 1 ? "current" : "todo";
        if (run.state?.preparation_complete && step >= 4) return "todo";
        if (this.reusedExport && step <= 3) return "skipped";
        const at = this.flowStep;
        if (step < at) return "done";
        if (step > at) return "todo";
        if (run.status === "failed") return "failed";
        if (run.status === "stopped") return "stopped";
        if (run.status === "needs_review") return "gate";
        return "current";
      },
      /** 步骤的简短状态文字，颜色之外始终有文字说明。
       * @param {number} step 1 到 5。
       * @returns {string} 状态文字。
       */
      stepText(step) {
        const state = this.stepState(step);
        if (state === 'skipped' && ['upload','annotate','review'].includes(this.page)) return '使用上传标签';
        if (step === 5 && state === "current" && ["completed", "stopped"].includes(this.agent?.status))
          return "可以导出";
        return {
          done: "已完成",
          current: this.agent ? "进行中" : "从这里开始",
          gate: "等你处理",
          todo: "未开始",
          skipped: "复用已审数据",
          failed: "失败",
          stopped: "已停止",
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
            dataset_copy: "跨项目复制数据集",
            import: "图片导入",
            image_recheck: "重新检查异常原图",
            analysis: "质量分析",
            version: "版本封存",
            auto_annotate: "候选推理",
            cvat_push: "发送 CVAT",
            cvat_pull: "回传人工审核",
            yolo_export: "封存训练数据",
          }[kind] || kind
        );
      },
      taskProgressText(task) {
        const detail = task?.progress_detail;
        if (!detail) return "";
        const count = `${detail.estimated ? "约 " : ""}${detail.completed_images} / ${detail.total_images} 张`;
        const stages = {
          annotation_queued: detail.independent_queue ? "等待自动标注 Worker 接单；与训练使用独立队列" : "仍在旧计算队列排队；这条任务保留原队列位置",
          loading_annotation_models: `正在加载自动标注模型 · ${detail.device || "执行设备待确认"}`,
          annotating_images: `正在自动标注 · ${count} · ${detail.device || "执行设备待确认"}`,
          preparing_source: "正在读取数据包结构与来源配置",
          extracting_archive: `正在解包 · ${detail.completed_files||0} / ${detail.total_files||0} 个文件`,
          normalizing_labels: `正在适配图片编码并检查标签 · 已处理 ${detail.completed_images||0} 张`,
          parsing_labels: `正在解析 YOLO 类别与标签 · 共 ${detail.total_images||0} 张`,
          checking_images: `检查图片编码、预览与标注 · ${count}`,
          copying_images: `复制已审核图片与标签 · ${count}`,
          preparing_images: `准备 CVAT 审核图片 · ${count}${detail.prepared_bytes ? ` · 已生成 ${this.bytes(detail.prepared_bytes)}` : ""}`,
          preparing_or_uploading: "图片准备即将完成，等待 CVAT 接收；尚未完成上传",
          connecting_cvat: "审核图片已准备完成，正在连接 CVAT",
          uploading_images: "审核图片已准备完成，正在上传并由 CVAT 创建审核任务",
          matching_frames: "CVAT 已接收图片，正在核对图片对应关系",
          uploading_annotations: "正在向 CVAT 写入标注框与类别",
          reading_review: "正在读取 CVAT 验收状态和审核结果",
          validating_review: "正在核对图片、类别与标注框，并保存审核记录",
          export_working: "正在核验图片、生成 YOLO 工件或压缩归档；当前任务未提供逐项进度",
          export_verifying_review: "正在复核已保存的审核结果与 CVAT 标注是否一致",
          export_preparing_images: `正在逐张核验原图与审核图片 · ${count}`,
          export_writing_dataset: `图片核验完成，正在生成 YOLO 数据与标签 · 共 ${detail.total_images} 张`,
          export_hashing: `正在校验工件指纹 · ${detail.completed_files || 0} / ${detail.total_files || 0} 个文件`,
          export_archiving: `正在压缩可下载数据包${detail.archive_bytes ? ` · 已写入 ${this.bytes(detail.archive_bytes)}` : ""}`,
          export_verifying_archive: "正在校验数据包并完成封存",
          export_ready: "训练数据已准备完成，等待你选择并开始训练",
          analyzing_dataset: "正在检查重复图片与采集分组",
          validating_version: `正在核验并封存数据版本 · ${count}`,
          writing_version: "正在保存版本清单与固定划分",
        };
        const text = stages[detail.stage] || "";
        const elapsed = detail.elapsed_seconds;
        return text && elapsed != null ? `${text} · 已用时 ${Math.floor(elapsed / 60)} 分 ${elapsed % 60} 秒` : text;
      },
      evaluationProgress(task) {
        const progress = task.summary?.progress;
        if (!progress) return this.status(task.status);
        return ({verifying_inputs:'核验已审数据与权重',validating:`验证图片 · ${progress.completed_batches||0} / ${progress.total_batches||'—'} 个批次`,analyzing_errors:'定位误检和漏检，生成结果图'})[progress.stage] || this.status(task.status);
      },
      modelExportProgress(task) {
        if (task.status === 'completed') return task.verification?.runtime ? `${task.verification.runtime} 加载通过` : '已完成';
        return task.error || ({verifying_weights:'核验原始权重',converting:'转换部署格式',verifying_runtime:'验证部署文件可加载与推理'})[task.verification?.progress] || this.status(task.status);
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
       * @param {string|object} result 已执行工具返回的结果。
       * @returns {string} 摘要。
       */
      toolDetail(tool, args, result) {
        if (tool === "Skill" || tool === "SkillRead") return [args.skill_name, args.file_name].filter(Boolean).join(" · ");
        if (tool === "answer_operator") return args.answer || "";
        if (tool === "inspect_training_parameters") return (args.names || []).join("、") || "查询参数范围与来源";
        if (tool === "read_dataset_examples") return [args.split === "val" ? "验证集" : "训练集", args.class_name, (args.image_ids || []).join("、")].filter(Boolean).join(" · ");
        if (tool === "read_validation_errors") return [args.class_name || "全部类别", args.small_only ? "小目标漏检" : ({ fn: "漏检", fp: "误检", any: "漏检与误检" })[args.error_kind || "fn"], args.image_id].filter(Boolean).join(" · ");
        if (tool === "compare_experiments") return [args.baseline_experiment_id, args.candidate_experiment_id].map((id) => this.entityLabel("experiment", id)).join(" → ");
        if (tool === "preview_experiment") {
          let status = "";
          try {
            const outcome = typeof result === "string" ? JSON.parse(result) : result;
            status = outcome?.valid === false ? "配置需修改" : outcome?.can_start === true ? "预检通过" : outcome?.valid === true ? "配置有效，等待开训条件" : "";
          } catch (_) {}
          return [status, this.paramText(args.parameters)].filter(Boolean).join(" · ") || "检查参数、数据与预算";
        }
        if (tool === "get_experiment") return [this.entityLabel("experiment", args.experiment_id), ({ overview: "概览", parameters: "参数", curve: "逐轮指标", evaluation: "验证结果", training: "训练尝试" })[args.section || "overview"]].join(" · ");
        if (tool === "start_experiment") {
          const changes = this.paramText(args.parameters);
          if (changes) return changes;
          try {
            const outcome = typeof result === "string" ? JSON.parse(result) : result;
            if (outcome?.iteration === 0) return "首轮采用平台默认值及已填写参数";
            if (outcome?.iteration > 0) return args.parent_experiment_id ? "沿用所选父实验参数" : "沿用上一轮参数";
          } catch (_) {}
          return "未显式调整参数，实际配置见实验详情";
        }
        if (tool === "read_chart")
          return `${args.source === "training" ? "训练" : "评估"} · ${(args.charts || []).map((x) => chartNames[x] || x).join("、")}`;
        if (tool === "sleep") return `${args.minutes} 分钟`;
        if (tool === "export_model")
          return `${this.formatName(args.format)} · ${args.kind === "last" ? "最后轮" : "最佳轮"}权重`;
        if (tool === "finish") return args.summary || "";
        if (tool === "request_human_review") return (args.data_actions || []).join("；");
        return args.experiment_id ? this.entityLabel("experiment", args.experiment_id) : "";
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
          changes.push(`数据 → ${this.entityLabel("export", item.plan.dataset_export_id)}`);
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
            context_compacted: "已压缩早期对话",
            context_window_rejected: "调整上下文窗口",
            operator_instruction_set: "保存持续指令",
            operator_instruction_revoked: "撤销持续指令",
            woke: "醒来",
            validation_observed: "验证结论",
            experiment_created: "新建实验",
            experiment_abandoned: "放弃实验",
            checkpoint_resumed: "从检查点继续",
            wake_requested: "手动唤醒 Agent",
            operator_answer: "Agent 回答",
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
      eventDetail(event) {
        const payload = event.payload || {};
        if (event.event_type === "wake_requested") return payload.question || "未附带问题，立即检查最新训练状态";
        if (event.event_type === "operator_answer") return payload.answer;
        if (event.event_type === "tool_call") return [this.toolDetail(payload.tool, payload.arguments || {}, payload.result), payload.arguments?.reason, payload.ok === false ? "执行被拒绝，展开原始记录查看原因" : ""].filter(Boolean).join("\n");
        if (event.event_type === "validation_observed") return `第 ${payload.iteration + 1} 轮验证完成：mAP50–95 ${this.metric(payload.summary?.map50_95)}，mAP50 ${this.metric(payload.summary?.map50)}`;
        if (event.event_type === "sealed") return payload.best_model ? this.entityLabel("model", payload.best_model.model_id) : "报告已封存，没有有效候选模型";
        return payload.content || payload.reason || payload.message || (payload.experiment_id ? this.entityLabel("experiment", payload.experiment_id) : "");
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
        return this.projects.find(project => project.id === id)?.name || "历史项目";
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
        if (['annotate','review'].includes(this.page)) return this.scopeDataset ? `${this.scopeDataset.name} · ${this.sourceName(this.scopeVersion?.source)} · ${this.dataState(this.scopeVersion?.status)}` : '先选择要处理的数据集，每份数据独立标注和审核';
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
      get importedAnnotation() {
        const run = this.currentAnnotation;
        return Boolean(run && (run.model_version?.startsWith("external") || run.parameters?.review_attribution || this.versions.find(item => item.id === run.version_id)?.policy?.preserve_source_splits));
      },
      get cvatEmbedUrl() {
        const run = this.currentAnnotation;
        return run?.cvat_task_id && this.cvatSessionKey === `${run.id}:${run.cvat_task_id}` ? this.cvatSessionUrl : "";
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
        const prepared = this.preparedModels;
        return [...new Set([...prepared, ...(this.trainingReadback && this.trainForm.params.model ? [this.trainForm.params.model] : [])])];
      },
      get trainingOptimizerChoices() {
        return [...new Set([...(this.searchSpace.optimizer || this.trainingOptimizers), ...(this.trainingReadback && this.trainForm.params.optimizer ? [this.trainForm.params.optimizer] : [])])];
      },
      get dialogTitle() {
        return (
          {
            project: "创建项目",
            dataset: "新建数据集",
            copyDataset: "复制其他项目的数据集",
            version: "封存分组版本",
            schema: "新建类别版本",
            annotate: "生成候选标注",
            sync: "确认并回传审核结果",
            experiment: "手动创建实验",
            resumeAgent: this.reviewConfirmation ? "确认并回传审核结果" : "人工处理与恢复",
            publish: "确认生产发布",
            annotationImage: "标注只读预览",
            deleteAsset: this.form.kind === "dataset" ? "删除数据集" : "删除已上传图片",
          }[this.modal] || ""
        );
      },
      get reviewConfirmation() {
        return this.modal === "sync" || (this.modal === "resumeAgent" && ["cvat_review", "annotation_validation"].includes(this.agent?.phase));
      },
      get reviewConfirmationOutcome() {
        return this.modal === "resumeAgent"
          ? "提交后系统会核验并回传审核结果，成功后自动封存为训练数据。数据就绪后，仍由你到训练页确认并开始训练。"
          : "提交后系统会核验并回传审核结果。回传成功后，请再点击“封存为训练数据”；封存完成才可在训练页选择。";
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
        if (run.state?.preparation_complete) return 4;
        if (run.status === 'stopped') return run.best_model_id ? 5 : run.dataset_export_id ? 4 : PHASE_STEP[run.phase] || 2;
        if (run.status === 'completed') return 5;
        return PHASE_STEP[run.phase] || 2;
      },
      get reusedExport() {
        // 从已审数据直接训练的流程没有自己的标注批次。
        return Boolean(this.agent?.dataset_export_id && !this.agent?.state?.annotation_run_id);
      },
      get showTrainForm() {
        return this.trainNew || !this.agent || this.agent.state?.preparation_complete || (this.agent.phase === "training_setup" && this.agent.status === "needs_review");
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
        return iteration ? `${this.entityLabel("experiment", iteration.experiment_id)} · Agent 已选定` : "等待 Agent 规划首轮参数；已填写值保留显示";
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
      get isAgentSleeping() {
        return this.agent?.status === "waiting" && this.agent.phase === "agent_sleep" && this.agent.state?.waiting?.kind === "sleep";
      },
      get wakeRequestPending() {
        if (!this.isAgentSleeping) return false;
        const until = this.agent.state.waiting.until;
        return this.agentWakeAcceptedUntil === until || this.operatorMessages.some(item => item.sleep_until === until);
      },
      get canWakeAgent() {
        return this.isAgentSleeping && !this.wakeRequestPending && !this.agent.pause_requested && !this.agent.stop_requested
          && (this.agent.state.model_calls || 0) < (this.agent.policy.max_model_calls || 120);
      },
      get agentChatHint() {
        if (this.wakeRequestPending) return "已请求唤醒，请等待 Agent 处理。当前训练继续运行。";
        if (this.isAgentSleeping && (this.agent.state.model_calls || 0) >= (this.agent.policy.max_model_calls || 120)) return "模型调用预算已耗尽，无法继续唤醒。";
        if (this.canWakeAgent) return "可提问或说明后续要求，例如“以后每次休眠10分钟”。Agent 保存的持续指令显示在上方，可随时撤销。模型调用计入本次预算。";
        return "当前不在休眠状态。可查看对话记录，待 Agent 再次休眠时唤醒提问。";
      },
      get agentStateText() {
        const run = this.agent;
        if (!run) return "";
        const waiting = run.state?.waiting || {};
        if (run.status === "failed") return "运行失败，等待人工恢复";
        if (run.status === "paused") return "已暂停";
        if (run.status === "canceling") return "正在停止";
        if (run.status === "stopped") return "已停止";
        if (run.state?.preparation_complete) return "数据准备已完成，训练已转到独立流程";
        if (run.status === "completed") return "已完成，可以导出模型";
        if (run.status === "needs_review")
          return (
            {
              cvat_review: "等你在 CVAT 审核",
              training_setup: "等你确认参数并开始训练",
              agent_review: "Agent 请求你处理",
            }[run.phase] || "等待人工处理"
          );
        if (run.phase === "cvat_upload" && this.cvatTransfer?.progress_detail?.stage === "preparing_images")
          return "正在准备 CVAT 审核图片";
        if (run.phase === "agent_sleep" && waiting.kind === "sleep")
          return this.wakeRequestPending ? "已请求唤醒，等待 Agent 响应" : `休眠中，预计 ${this.clock(waiting.until)} 醒来`;
        if (waiting.kind === "finish_tasks" && run.status === "waiting") return "正在收尾：等待评估与导出结束";
        return this.phase(run.phase);
      },
      get agentStateDetail() {
        const run = this.agent;
        if (!run) return "";
        if (run.phase === "cvat_upload" && this.cvatTransfer) return this.taskProgressText(this.cvatTransfer);
        if (["annotation_validation", "reviewed_export"].includes(run.phase) && this.reviewTransfer)
          return this.taskProgressText(this.reviewTransfer) || this.reviewTransfer.error || "等待后台任务执行";
        const waiting = run.state?.waiting || {};
        if (waiting.kind === "finish_tasks" && run.status === "waiting") return "已有训练结果保留；后台任务完成校验后封存报告，可随时停止整个流程。";
        if (run.phase === "agent_sleep" && waiting.kind === "sleep")
          return this.wakeRequestPending ? "唤醒请求已保存，Agent 将读取最新训练状态并处理附带问题。" : `${waiting.reason}。训练、评估或导出结束时会提前醒来，也可以手动唤醒。`;
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
      get cvatTransfer() {
        return this.annotationTasks.find(task => task.kind === "cvat_push") || null;
      },
      get reviewTransfer() {
        const kind = this.agent?.phase === "annotation_validation" ? "cvat_pull" : "yolo_export";
        const waitingId = this.agent?.state?.waiting?.task_id;
        return this.annotationTasks.find(task => task.id === waitingId && task.kind === kind)
          || this.annotationTasks.find(task => task.kind === kind) || null;
      },
      get pipelineStages() {
        const run = this.agent;
        if (!run) return [];
        const at = PHASE_ORDER.indexOf(run.phase);
        const finished = run.status === "completed";
        const batch = this.annotations.find((x) => x.id === run.state?.annotation_run_id);
        const analysis = run.state?.dataset?.analysis;
        const stages = [
          { key: "inspect", name: "检查数据", phases: ["dataset_inspector", "data_quality_check"], detail: analysis ? `有效图片 ${analysis.valid_images} 张` : "" },
          { key: "split", name: "划分 train / val / test 并封存", phases: ["dataset_split"], detail: run.dataset_version_id ? this.entityLabel("version", run.dataset_version_id) : "" },
          {
            key: "annotate",
            name: "按所选标注来源生成候选框",
            phases: ["auto_annotation", "annotation_quality"],
            detail: batch ? `${batch.processed} / ${batch.total} 张图` : "",
            done: batch?.processed,
            total: batch && ["queued", "running"].includes(batch.status) ? batch.total : 0,
          },
          { key: "cvat", name: "准备审核图片并发送到 CVAT", phases: ["cvat_upload"], detail: batch?.cvat_task_id ? `CVAT 任务 #${batch.cvat_task_id}` : this.taskProgressText(this.cvatTransfer), done: this.cvatTransfer?.processed || 0, total: this.cvatTransfer && ACTIVE.includes(this.cvatTransfer.status) ? this.cvatTransfer.total : 0 },
        ];
        return stages.map((stage) => {
          const first = PHASE_ORDER.indexOf(stage.phases[0]);
          const last = PHASE_ORDER.indexOf(stage.phases.at(-1));
          const state = this.reusedExport
            ? "skipped"
            : finished || at > last
              ? "done"
              : at >= first
                ? run.status === "stopped"
                  ? "stopped"
                  : run.status === "failed"
                  ? "failed"
                  : "current"
                : "todo";
          return { ...stage, state, label: { done: "完成", current: "进行中", stopped: "已停止", failed: "失败", todo: "等待", skipped: "已复用" }[state] };
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
              detail: p.ok ? this.toolDetail(p.tool, args, p.result) : `被拒绝：${error}`,
              reason: args.reason,
              tone: p.ok ? (p.tool === "sleep" ? "muted" : "info") : "danger",
              stage: ({ Skill: "知识", SkillRead: "知识", inspect_training_parameters: "分析", read_dataset_examples: "分析", read_validation_errors: "分析", compare_experiments: "分析", preview_experiment: "决策", start_experiment: "训练", stop_training: "训练", resume_training: "训练", evaluate: "评估", read_chart: "分析", read_confusion_matrix: "分析", get_experiment: "分析", read_training_log: "分析", suggest_parameters: "决策", sleep: "等待", finish: "收尾", export_model: "导出" })[p.tool] || "执行",
            };
          } else if (event.event_type === "model_decision" && p.content)
            item = { title: "Agent 说明", detail: p.content, tone: "muted" };
          else if (event.event_type === "woke") item = { title: "醒来", detail: p.reason, tone: "muted" };
          else if (event.event_type === "wake_requested") item = { title: "你唤醒了 Agent", detail: p.question || "检查最新训练状态", tone: "info" };
          else if (event.event_type === "operator_answer") item = { title: "Agent 回答", detail: p.answer, tone: "info" };
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
          else if (event.event_type === "data_prepared")
            item = { title: "数据准备完成", detail: p.message, tone: "success" };
          else if (event.event_type === "failed")
            item = { title: `${this.phase(event.node)}失败`, detail: p.message, tone: "danger" };
          else if (event.event_type === "image_request_rejected")
            item = { title: "模型图像请求被拒绝", detail: p.action, tone: "warning" };
          else if (event.event_type === "finish_task_unavailable")
            item = { title: "收尾任务未执行", detail: p.message, tone: "warning" };
          else if (event.event_type === "sealed")
            item = { title: "运行报告已封存", detail: p.best_model ? this.entityLabel("model", p.best_model.model_id) : "没有有效候选", tone: "success" };
          else if (event.node === "operator" && ["pause", "resume", "stop"].includes(event.event_type))
            item = { title: { pause: "操作者暂停", resume: "操作者恢复", stop: "操作者停止" }[event.event_type], detail: p.reason || "", tone: "muted" };
          if (item) items.push({ id: event.id, time: event.created_at, stage: "流程", ...item, preview: this.textPreview(item.detail), reasonPreview: this.textPreview(item.reason, 90), raw: JSON.stringify(p, null, 2) });
        }
        return items.reverse();
      },
      get trainingStages() {
        const iteration = this.latestTrainingIteration;
        const experiment = this.experiments.find(item => item.id === iteration?.experiment_id);
        const terminated = ["stopped", "failed", "paused", "completed"].includes(this.agent?.status);
        const trained = experiment?.status === "completed";
        const evaluated = iteration?.conclusion?.map50_95 != null;
        const index = !iteration ? 0 : !trained ? 1 : !evaluated ? 2 : 3;
        return ["分析与规划", "训练模型", "验证与对比", "决定下一步"].map((name, at) => ({ name, state: at < index ? "done" : at === index ? (terminated ? this.agent.status : "current") : "todo", label: at < index ? "已完成" : at === index ? (terminated ? this.status(this.agent.status) : "进行中") : "等待" }));
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
