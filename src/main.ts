#!/usr/bin/env node
import { execFile } from "node:child_process";
import process from "node:process";
import { promisify } from "node:util";
import blessed from "blessed";
import contrib from "blessed-contrib";

const execFileAsync = promisify(execFile);

const DEFAULT_REGIONS = [
  "us-east-1",
  "us-east-2",
  "us-west-2",
  "eu-west-1",
  "eu-central-1",
  "ap-southeast-1",
  "ap-northeast-1",
];

const SORT_MODES = ["region_gpu", "best"] as const;

type SortMode = (typeof SORT_MODES)[number];

type Args = {
  profile?: string;
  regions: string;
  gpuModels?: string;
  instanceTypes?: string;
  catalogRegion: string;
  interval: number;
  targetInstances: number;
  maxRows: number;
  sort: SortMode;
  once: boolean;
  historyHours: number;
  devFast: boolean;
};

type InstanceSpec = {
  instanceType: string;
  gpuModel: string;
  gpuCount: number;
  gpuMemoryGiB: number;
  totalGpuMemoryGiB: number;
  vcpus: number;
  memoryGiB: number;
  networkPerformance: string;
};

type InstanceOption = {
  instanceType: string;
  gpuCount: number;
  gpuMemoryGiB: number;
  azCount: number;
  spotInstance?: number;
  spotGpu?: number;
  onDemandInstance?: number;
  onDemandGpu?: number;
  vcpus: number;
  memoryGiB: number;
  networkPerformance: string;
};

type ModelRegionRow = {
  region: string;
  gpuModel: string;
  options: InstanceOption[];
  azCount: number;
  totalAzCount?: number;
  typeCount: number;
  gpuCountRange: [number, number];
  gpuMemoryGiB: number;
  bestType: string;
  bestSpotInstance?: number;
  bestSpotGpu?: number;
  bestOnDemandInstance?: number;
  bestOnDemandGpu?: number;
  spotScore?: number;
};

type RefreshSnapshot = {
  rows: ModelRegionRow[];
  lastRefreshAt?: Date;
  errors: string[];
};

type TimePoint = {
  at: Date;
  value: number;
};

type HistorySeries = {
  title: string;
  points: TimePoint[];
};

type HistoryCacheEntry = {
  fetchedAt: number;
  series: HistorySeries[];
};

type ActivePane = "regions" | "selection";

const GPU_MODEL_RULES: Array<[string, string]> = [
  ["B200", "b200"],
  ["B100", "b100"],
  ["H200", "h200"],
  ["H100", "h100"],
  ["A100", "a100"],
  ["A10G", "a10g"],
  ["L40S", "l40s"],
  ["L4", "l4"],
  ["T4G", "t4g"],
  ["T4", "t4"],
  ["V100", "v100"],
  ["A16", "a16"],
  ["M60", "m60"],
  ["K80", "k80"],
  ["K520", "k520"],
];

const DEFAULT_TRAINING_GPU_MODELS = new Set([
  "B200",
  "B100",
  "H200",
  "H100",
  "A100",
  "A10G",
  "L40S",
  "L4",
  "V100",
]);

const GPU_GENERATION_ORDER = [
  "B200",
  "B100",
  "H200",
  "H100",
  "A100",
  "A10G",
  "L40S",
  "L4",
  "V100",
  "T4G",
  "T4",
  "A16",
  "M60",
  "K80",
  "K520",
] as const;

const REGION_SORT_ORDER = [
  "us-east-1",
  "us-east-2",
  "us-west-1",
  "us-west-2",
  "ca-central-1",
  "ca-west-1",
  "eu-west-1",
  "eu-west-2",
  "eu-west-3",
  "eu-central-1",
  "eu-central-2",
  "eu-north-1",
  "eu-south-2",
  "eu-south-1",
  "il-central-1",
  "me-south-1",
  "me-central-1",
  "af-south-1",
  "ap-east-1",
  "ap-east-2",
  "ap-south-1",
  "ap-south-2",
  "ap-southeast-1",
  "ap-southeast-2",
  "ap-southeast-3",
  "ap-southeast-4",
  "ap-southeast-5",
  "ap-southeast-7",
  "ap-northeast-1",
  "ap-northeast-2",
  "ap-northeast-3",
  "sa-east-1",
  "mx-central-1",
] as const;

const REGION_LOCATION_NAMES: Record<string, string> = {
  "af-south-1": "Africa (Cape Town)",
  "ap-east-1": "Asia Pacific (Hong Kong)",
  "ap-east-2": "Asia Pacific (Taipei)",
  "ap-northeast-1": "Asia Pacific (Tokyo)",
  "ap-northeast-2": "Asia Pacific (Seoul)",
  "ap-northeast-3": "Asia Pacific (Osaka)",
  "ap-south-1": "Asia Pacific (Mumbai)",
  "ap-south-2": "Asia Pacific (Hyderabad)",
  "ap-southeast-1": "Asia Pacific (Singapore)",
  "ap-southeast-2": "Asia Pacific (Sydney)",
  "ap-southeast-3": "Asia Pacific (Jakarta)",
  "ap-southeast-4": "Asia Pacific (Melbourne)",
  "ap-southeast-5": "Asia Pacific (Malaysia)",
  "ap-southeast-7": "Asia Pacific (Thailand)",
  "ca-central-1": "Canada (Central)",
  "ca-west-1": "Canada West (Calgary)",
  "eu-central-1": "EU (Frankfurt)",
  "eu-central-2": "Europe (Zurich)",
  "eu-north-1": "EU (Stockholm)",
  "eu-south-1": "Europe (Milan)",
  "eu-south-2": "Europe (Spain)",
  "eu-west-1": "EU (Ireland)",
  "eu-west-2": "EU (London)",
  "eu-west-3": "EU (Paris)",
  "il-central-1": "Israel (Tel Aviv)",
  "me-central-1": "Middle East (UAE)",
  "me-south-1": "Middle East (Bahrain)",
  "mx-central-1": "Mexico (Central)",
  "sa-east-1": "South America (Sao Paulo)",
  "us-east-1": "US East (N. Virginia)",
  "us-east-2": "US East (Ohio)",
  "us-west-1": "US West (N. California)",
  "us-west-2": "US West (Oregon)",
};

class AwsCliError extends Error {}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    regions: "default",
    catalogRegion: "us-east-1",
    interval: 20,
    targetInstances: 1,
    maxRows: 200,
    sort: "region_gpu",
    once: false,
    historyHours: 24,
    devFast: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--profile" && next) {
      args.profile = next;
      index += 1;
    } else if (arg === "--regions" && next) {
      args.regions = next;
      index += 1;
    } else if (arg === "--gpu-models" && next) {
      args.gpuModels = next;
      index += 1;
    } else if (arg === "--instance-types" && next) {
      args.instanceTypes = next;
      index += 1;
    } else if (arg === "--catalog-region" && next) {
      args.catalogRegion = next;
      index += 1;
    } else if (arg === "--interval" && next) {
      args.interval = Number(next);
      index += 1;
    } else if (arg === "--target-instances" && next) {
      args.targetInstances = Number(next);
      index += 1;
    } else if (arg === "--max-rows" && next) {
      args.maxRows = Number(next);
      index += 1;
    } else if (arg === "--sort" && next && SORT_MODES.includes(next as SortMode)) {
      args.sort = next as SortMode;
      index += 1;
    } else if (arg === "--history-hours" && next) {
      args.historyHours = Number(next);
      index += 1;
    } else if (arg === "--once") {
      args.once = true;
    } else if (arg === "--dev-fast") {
      args.devFast = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }
  if (args.devFast) {
    if (args.regions === "default") args.regions = "us-east-1";
    if (!args.gpuModels) args.gpuModels = "H100,A100,L4,L40S";
    if (args.historyHours === 24) args.historyHours = 6;
    if (args.interval === 20) args.interval = 60;
    if (args.maxRows === 200) args.maxRows = 40;
  }
  return args;
}

function printHelp(): void {
  process.stdout.write(`Usage: spotter [options]

Options:
  --profile NAME
  --regions default|all|REGION1,REGION2
  --gpu-models H100,A100,L40S
  --instance-types p5.4xlarge,p4d.24xlarge
  --catalog-region us-east-1
  --interval 20
  --target-instances 1
  --max-rows 200
  --history-hours 24
  --sort ${SORT_MODES.join("|")}
  --dev-fast
  --once
`);
}

function canonicalGpuModel(rawName: string): string {
  const upper = rawName.toUpperCase();
  for (const [display] of GPU_MODEL_RULES) {
    if (upper.includes(display)) {
      return display;
    }
  }
  return rawName.replace("NVIDIA", "").trim() || rawName;
}

function chunk<T>(values: T[], size: number): T[][] {
  const output: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    output.push(values.slice(index, index + size));
  }
  return output;
}

function formatMoney(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return "n/a";
  if (value >= 100) return `$${value.toFixed(0)}`;
  if (value >= 10) return `$${value.toFixed(2)}`;
  return `$${value.toFixed(3)}`;
}

function safeRatio(numerator?: number, denominator?: number): number | undefined {
  if (numerator === undefined || denominator === undefined || denominator <= 0) {
    return undefined;
  }
  return numerator / denominator;
}

function scoreBar(score?: number, width = 10): string {
  if (score === undefined) return "·".repeat(width);
  const filled = Math.max(0, Math.min(width, Math.round((score / 10) * width)));
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function compactRange(value: [number, number]): string {
  return value[0] === value[1] ? String(value[0]) : `${value[0]}-${value[1]}`;
}

function formatHourLabel(at: Date): string {
  const hour = at.getHours();
  const suffix = hour >= 12 ? "pm" : "am";
  const normalized = hour % 12 || 12;
  return `${normalized}${suffix}`;
}

function niceStep(rawStep: number): number {
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 1;
  const exponent = Math.floor(Math.log10(rawStep));
  const fraction = rawStep / 10 ** exponent;
  let niceFraction = 1;
  if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

function priceFromOffer(offer: any): number | undefined {
  const onDemand = offer?.terms?.OnDemand ?? {};
  for (const term of Object.values<any>(onDemand)) {
    for (const dimension of Object.values<any>(term?.priceDimensions ?? {})) {
      if (dimension?.unit !== "Hrs") continue;
      const usd = Number(dimension?.pricePerUnit?.USD);
      if (Number.isFinite(usd) && usd > 0) return usd;
    }
  }
  return undefined;
}

function stripTags(text: string): string {
  return text.replace(/\{\/?[^}]+\}/g, "");
}

function fitCell(value: string, width: number): string {
  if (width <= 0) return "";
  if (value.length === width) return value;
  if (value.length < width) return value.padEnd(width, " ");
  if (width === 1) return value[0];
  return `${value.slice(0, width - 1)}…`;
}

function fitTaggedCell(value: string, width: number): string {
  const visible = stripTags(value);
  if (visible.length <= width) {
    return value + " ".repeat(width - visible.length);
  }
  return fitCell(visible, width);
}

function mutedUnit(unit: string): string {
  return `{gray-fg}${unit}{/gray-fg}`;
}

function scoreColor(score?: number): string {
  if (score === undefined) return "gray";
  if (score >= 8) return "green";
  if (score >= 5) return "yellow";
  if (score >= 3) return "magenta";
  return "red";
}

function regionSortRank(region: string): number {
  const index = REGION_SORT_ORDER.indexOf(region as (typeof REGION_SORT_ORDER)[number]);
  return index >= 0 ? index : REGION_SORT_ORDER.length;
}

function gpuGenerationRank(gpuModel: string): number {
  const index = GPU_GENERATION_ORDER.indexOf(gpuModel as (typeof GPU_GENERATION_ORDER)[number]);
  return index >= 0 ? index : GPU_GENERATION_ORDER.length;
}

function formatAzCoverage(available: number, total?: number): string {
  if (total === undefined || total <= 0) return `${available}/?`;
  return `${available}/${total}`;
}

function comparableHourlyPrice(row: ModelRegionRow): number {
  return row.bestSpotInstance ?? row.bestOnDemandInstance ?? Number.POSITIVE_INFINITY;
}

function availabilitySignal(row: ModelRegionRow): number {
  const scorePart = Math.max(0, Math.min(1, (row.spotScore ?? 0) / 10));
  const azPart =
    row.totalAzCount && row.totalAzCount > 0
      ? Math.max(0, Math.min(1, row.azCount / row.totalAzCount))
      : Math.max(0, Math.min(1, row.azCount / 6));
  return scorePart * 0.7 + azPart * 0.3;
}

function generationSignal(row: ModelRegionRow): number {
  const generationCount = Math.max(1, GPU_GENERATION_ORDER.length - 1);
  const generationPart = 1 - gpuGenerationRank(row.gpuModel) / generationCount;
  const vramPart = Math.max(0, Math.min(1, row.gpuMemoryGiB / 192));
  return generationPart * 0.65 + vramPart * 0.35;
}

function proximitySignal(row: ModelRegionRow): number {
  const regionCount = Math.max(1, REGION_SORT_ORDER.length - 1);
  return 1 - Math.min(regionSortRank(row.region), regionCount) / regionCount;
}

function priceSignal(row: ModelRegionRow): number {
  const price = comparableHourlyPrice(row);
  if (!Number.isFinite(price) || price <= 0) return 0;
  return 1 / (1 + Math.log10(1 + price));
}

function bestHeuristicScore(row: ModelRegionRow): number {
  return (
    availabilitySignal(row) * 0.35 +
    generationSignal(row) * 0.25 +
    priceSignal(row) * 0.25 +
    proximitySignal(row) * 0.15
  );
}

function compactNetwork(value: string): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return "n/a";
  return normalized
    .replace(/^Up to /i, "<= ")
    .replace(/ Gigabit$/i, "G")
    .replace(/ Gigabit /gi, "G ")
    .replace(/ Gbps$/i, "G")
    .replace(/ Inferred$/i, "");
}

async function mapConcurrent<T, R>(
  values: T[],
  limit: number,
  task: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < values.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await task(values[currentIndex], currentIndex);
    }
  }

  await Promise.all(new Array(Math.max(1, Math.min(limit, values.length))).fill(null).map(() => worker()));
  return results;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

class AwsCli {
  constructor(private readonly profile?: string) {}

  async runJson(command: string[], region?: string): Promise<any> {
    const globalArgs = ["--output", "json", "--no-cli-pager"];
    if (this.profile) {
      globalArgs.push("--profile", this.profile);
    }
    if (region) {
      globalArgs.push("--region", region);
    }
    const fullCommand = [...globalArgs, ...command];
    for (let attempt = 0; attempt < 4; attempt += 1) {
      try {
        const result = await execFileAsync("aws", fullCommand, {
          env: process.env,
          maxBuffer: 20 * 1024 * 1024,
        });
        return result.stdout.trim() ? JSON.parse(result.stdout) : {};
      } catch (error: any) {
        const stderr = error?.stderr?.toString()?.trim() || error?.message || "unknown AWS CLI error";
        const throttled = /Throttling|Rate exceeded|TooManyRequests/i.test(stderr);
        if (!throttled || attempt >= 3) {
          throw new AwsCliError(stderr);
        }
        await sleep(250 * 2 ** attempt);
      }
    }
    throw new AwsCliError("unknown AWS CLI error");
  }
}

class TrackerData {
  private readonly aws: AwsCli;
  private readonly args: Args;
  private regions: string[] = [];
  private regionAzCounts = new Map<string, number>();
  private specs = new Map<string, InstanceSpec>();
  private specsByModel = new Map<string, InstanceSpec[]>();
  private offerings = new Map<string, Map<string, Set<string>>>();
  private onDemandPrices = new Map<string, Map<string, number>>();

  constructor(args: Args) {
    this.args = args;
    this.aws = new AwsCli(args.profile);
  }

  getRegions(): string[] {
    return this.regions;
  }

  async initialize(): Promise<string[]> {
    this.regions = await this.resolveRegions(this.args.regions);
    await this.loadCatalog();
    return this.refreshStaticData();
  }

  async refreshSnapshot(): Promise<RefreshSnapshot> {
    const [[spotPrices, spotPriceErrors], [spotScores, spotScoreErrors]] = await Promise.all([
      this.fetchSpotPrices(),
      this.args.devFast ? Promise.resolve<[Map<string, Map<string, number>>, string[]]>([new Map(), []]) : this.fetchSpotScores(),
    ]);
    const rows: ModelRegionRow[] = [];

    for (const region of this.regions) {
      const regionOfferings = this.offerings.get(region);
      if (!regionOfferings) continue;
      for (const [gpuModel, specs] of this.specsByModel.entries()) {
        const offeredSpecs = specs.filter((spec) => regionOfferings.has(spec.instanceType));
        if (offeredSpecs.length === 0) continue;

        const optionRows: InstanceOption[] = [];
        const azIds = new Set<string>();
        let bestSpotInstance: number | undefined;
        let bestSpotGpu: number | undefined;
        let bestOnDemandInstance: number | undefined;
        let bestOnDemandGpu: number | undefined;
        let bestType = offeredSpecs[0].instanceType;

        for (const spec of offeredSpecs) {
          const offeringAzs = regionOfferings.get(spec.instanceType) ?? new Set<string>();
          offeringAzs.forEach((az) => azIds.add(az));
          const spotInstance = spotPrices.get(region)?.get(spec.instanceType);
          const onDemandInstance = this.onDemandPrices.get(region)?.get(spec.instanceType);
          const option: InstanceOption = {
            instanceType: spec.instanceType,
            gpuCount: spec.gpuCount,
            gpuMemoryGiB: spec.gpuMemoryGiB,
            azCount: offeringAzs.size,
            spotInstance,
            spotGpu: safeRatio(spotInstance, spec.gpuCount),
            onDemandInstance,
            onDemandGpu: safeRatio(onDemandInstance, spec.gpuCount),
            vcpus: spec.vcpus,
            memoryGiB: spec.memoryGiB,
            networkPerformance: spec.networkPerformance,
          };
          optionRows.push(option);

          if (option.spotGpu !== undefined && (bestSpotGpu === undefined || option.spotGpu < bestSpotGpu)) {
            bestSpotGpu = option.spotGpu;
          }
          if (
            option.spotInstance !== undefined &&
            (bestSpotInstance === undefined || option.spotInstance < bestSpotInstance)
          ) {
            bestSpotInstance = option.spotInstance;
            bestType = option.instanceType;
          }
          if (
            option.onDemandGpu !== undefined &&
            (bestOnDemandGpu === undefined || option.onDemandGpu < bestOnDemandGpu)
          ) {
            bestOnDemandGpu = option.onDemandGpu;
          }
          if (
            option.onDemandInstance !== undefined &&
            (bestOnDemandInstance === undefined || option.onDemandInstance < bestOnDemandInstance)
          ) {
            bestOnDemandInstance = option.onDemandInstance;
            if (bestSpotInstance === undefined) bestType = option.instanceType;
          }
        }

        optionRows.sort((left, right) => {
          const leftSpot = left.spotGpu ?? Number.POSITIVE_INFINITY;
          const rightSpot = right.spotGpu ?? Number.POSITIVE_INFINITY;
          if (leftSpot !== rightSpot) return leftSpot - rightSpot;
          const leftOD = left.onDemandGpu ?? Number.POSITIVE_INFINITY;
          const rightOD = right.onDemandGpu ?? Number.POSITIVE_INFINITY;
          if (leftOD !== rightOD) return leftOD - rightOD;
          return left.instanceType.localeCompare(right.instanceType);
        });

        rows.push({
          region,
          gpuModel,
          options: optionRows,
          azCount: azIds.size,
          totalAzCount: this.regionAzCounts.get(region),
          typeCount: offeredSpecs.length,
          gpuCountRange: [
            Math.min(...offeredSpecs.map((spec) => spec.gpuCount)),
            Math.max(...offeredSpecs.map((spec) => spec.gpuCount)),
          ],
          gpuMemoryGiB: Math.max(...offeredSpecs.map((spec) => spec.gpuMemoryGiB)),
          bestType,
          bestSpotInstance,
          bestSpotGpu,
          bestOnDemandInstance,
          bestOnDemandGpu,
          spotScore: spotScores.get(region)?.get(gpuModel),
        });
      }
    }

    return {
      rows: this.sortRows(rows, this.args.sort),
      lastRefreshAt: new Date(),
      errors: [...spotPriceErrors, ...spotScoreErrors],
    };
  }

  sortRows(rows: ModelRegionRow[], mode: SortMode): ModelRegionRow[] {
    const noneLast = (value: number | undefined, fallback: number): number => (value === undefined ? fallback : value);
    const sorted = [...rows];
    sorted.sort((left, right) => {
      const stableRegionGpuOrder =
        regionSortRank(left.region) - regionSortRank(right.region) ||
        left.region.localeCompare(right.region) ||
        gpuGenerationRank(left.gpuModel) - gpuGenerationRank(right.gpuModel) ||
        left.gpuModel.localeCompare(right.gpuModel);
      let score = 0;
      if (mode === "region_gpu") {
        score = stableRegionGpuOrder;
      } else {
        score =
          bestHeuristicScore(right) - bestHeuristicScore(left) ||
          noneLast(right.spotScore, -1) - noneLast(left.spotScore, -1) ||
          generationSignal(right) - generationSignal(left) ||
          comparableHourlyPrice(left) - comparableHourlyPrice(right);
      }
      return score || stableRegionGpuOrder;
    });
    return sorted;
  }

  async fetchHistory(row: ModelRegionRow, hours: number): Promise<HistorySeries[]> {
    const types = row.options.slice(0, this.args.devFast ? 2 : 3).map((option) => option.instanceType);
    if (types.length === 0) return [];
    const startTime = new Date(Date.now() - hours * 3600 * 1000).toISOString();
    const response = await this.aws.runJson(
      [
        "ec2",
        "describe-spot-price-history",
        "--product-descriptions",
        "Linux/UNIX",
        "--start-time",
        startTime,
        "--instance-types",
        ...types,
      ],
      row.region,
    );

    const buckets = new Map<string, Map<string, number>>();
    for (const entry of response?.SpotPriceHistory ?? []) {
      const instanceType = String(entry.InstanceType);
      if (!types.includes(instanceType)) continue;
      const option = row.options.find((candidate) => candidate.instanceType === instanceType);
      if (!option) continue;
      const at = new Date(entry.Timestamp);
      const bucketLabel = new Date(
        Date.UTC(at.getUTCFullYear(), at.getUTCMonth(), at.getUTCDate(), at.getUTCHours()),
      ).toISOString();
      const pricePerGpu = safeRatio(Number(entry.SpotPrice), option.gpuCount);
      if (pricePerGpu === undefined) continue;
      if (!buckets.has(instanceType)) buckets.set(instanceType, new Map());
      const current = buckets.get(instanceType)!;
      const existing = current.get(bucketLabel);
      if (existing === undefined || pricePerGpu < existing) {
        current.set(bucketLabel, pricePerGpu);
      }
    }

    const series: HistorySeries[] = [];
    for (const instanceType of types) {
      const bucketMap = buckets.get(instanceType);
      if (!bucketMap || bucketMap.size === 0) continue;
      const points = [...bucketMap.entries()]
        .sort((left, right) => left[0].localeCompare(right[0]))
        .map(([label, value]) => ({ at: new Date(label), value }));
      series.push({ title: instanceType, points });
    }
    return series;
  }

  private async resolveRegions(requested: string): Promise<string[]> {
    if (requested === "default") return [...DEFAULT_REGIONS];
    if (requested !== "all") {
      return requested
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
    }
    try {
      const response = await this.aws.runJson(["ec2", "describe-regions", "--all-regions"], this.args.catalogRegion);
      return (response?.Regions ?? [])
        .filter((region: any) => [undefined, "opt-in-not-required", "opted-in"].includes(region.OptInStatus))
        .map((region: any) => String(region.RegionName))
        .sort();
    } catch {
      return [...DEFAULT_REGIONS];
    }
  }

  private async loadCatalog(): Promise<void> {
    const requestedTypes = new Set(
      (this.args.instanceTypes ?? "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    const requestedModels =
      this.args.gpuModels && this.args.gpuModels.trim()
        ? new Set(
            this.args.gpuModels
              .split(",")
              .map((value) => value.trim().toUpperCase())
              .filter(Boolean),
          )
        : DEFAULT_TRAINING_GPU_MODELS;
    const response = await this.aws.runJson(["ec2", "describe-instance-types"], this.args.catalogRegion);
    for (const item of response?.InstanceTypes ?? []) {
      const gpus = item?.GpuInfo?.Gpus ?? [];
      if (gpus.length === 0) continue;
      if (gpus.some((gpu: any) => String(gpu.Manufacturer ?? "").toLowerCase() !== "nvidia")) continue;
      const gpuModel = canonicalGpuModel(String(gpus[0].Name ?? "Unknown"));
      const instanceType = String(item.InstanceType);
      if (requestedTypes.size > 0 && !requestedTypes.has(instanceType)) continue;
      if (requestedModels.size > 0 && !requestedModels.has(gpuModel.toUpperCase())) continue;
      const totalGpuCount = gpus.reduce((sum: number, gpu: any) => sum + Number(gpu.Count ?? 0), 0);
      const totalGpuMemoryGiB = Number(item?.GpuInfo?.TotalGpuMemoryInMiB ?? 0) / 1024;
      const spec: InstanceSpec = {
        instanceType,
        gpuModel,
        gpuCount: totalGpuCount,
        gpuMemoryGiB: totalGpuCount > 0 ? totalGpuMemoryGiB / totalGpuCount : 0,
        totalGpuMemoryGiB,
        vcpus: Number(item?.VCpuInfo?.DefaultVCpus ?? 0),
        memoryGiB: Number(item?.MemoryInfo?.SizeInMiB ?? 0) / 1024,
        networkPerformance: String(item?.NetworkInfo?.NetworkPerformance ?? ""),
      };
      this.specs.set(instanceType, spec);
      if (!this.specsByModel.has(gpuModel)) this.specsByModel.set(gpuModel, []);
      this.specsByModel.get(gpuModel)!.push(spec);
    }
    if (this.specs.size === 0) {
      throw new AwsCliError("No NVIDIA GPU instance types matched the current filters.");
    }
  }

  private async refreshStaticData(): Promise<string[]> {
    const errors: string[] = [];
    this.offerings = new Map();
    this.onDemandPrices = new Map();
    this.regionAzCounts = new Map();
    const instanceTypes = [...this.specs.keys()].sort();

    await mapConcurrent(this.regions, 4, async (region) => {
      try {
        const response = await this.aws.runJson(["ec2", "describe-availability-zones"], region);
        const count = (response?.AvailabilityZones ?? []).filter((zone: any) => {
          const state = String(zone.State ?? "");
          const zoneType = String(zone.ZoneType ?? "availability-zone");
          return state === "available" && zoneType === "availability-zone";
        }).length;
        if (count > 0) {
          this.regionAzCounts.set(region, count);
        }
      } catch (error: any) {
        errors.push(`${region} azs: ${String(error.message ?? error)}`);
      }
    });

    await mapConcurrent(this.regions, 4, async (region) => {
      const regionOfferings = new Map<string, Set<string>>();
      this.offerings.set(region, regionOfferings);
      try {
        await mapConcurrent(chunk(instanceTypes, 100), 4, async (part) => {
          const filters = `Name=instance-type,Values=${part.join(",")}`;
          const response = await this.aws.runJson(
            ["ec2", "describe-instance-type-offerings", "--location-type", "availability-zone-id", "--filters", filters],
            region,
          );
          for (const offering of response?.InstanceTypeOfferings ?? []) {
            const instanceType = String(offering.InstanceType);
            const az = String(offering.Location);
            if (!regionOfferings.has(instanceType)) regionOfferings.set(instanceType, new Set());
            regionOfferings.get(instanceType)!.add(az);
          }
        });
      } catch (error: any) {
        errors.push(`${region} offerings: ${String(error.message ?? error)}`);
      }
    });

    if (this.args.devFast) {
      return errors;
    }

    await mapConcurrent(this.regions, 3, async (region) => {
      const offeredTypes = [...(this.offerings.get(region)?.keys() ?? [])];
      this.onDemandPrices.set(region, new Map());
      const location = REGION_LOCATION_NAMES[region];
      if (!location) return;
      await mapConcurrent(offeredTypes, 6, async (instanceType) => {
        try {
          const response = await this.aws.runJson(
            [
              "pricing",
              "get-products",
              "--service-code",
              "AmazonEC2",
              "--filters",
              `Type=TERM_MATCH,Field=location,Value=${location}`,
              "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
              "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
              "Type=TERM_MATCH,Field=tenancy,Value=Shared",
              "Type=TERM_MATCH,Field=capacitystatus,Value=Used",
              "Type=TERM_MATCH,Field=licenseModel,Value=No License required",
              "Type=TERM_MATCH,Field=marketoption,Value=OnDemand",
              `Type=TERM_MATCH,Field=instanceType,Value=${instanceType}`,
            ],
            "us-east-1",
          );
          for (const rawOffer of response?.PriceList ?? []) {
            const price = priceFromOffer(JSON.parse(rawOffer));
            if (price !== undefined) {
              this.onDemandPrices.get(region)!.set(instanceType, price);
              break;
            }
          }
        } catch (error: any) {
          errors.push(`${region} pricing ${instanceType}: ${String(error.message ?? error)}`);
        }
      });
    });
    return errors;
  }

  private async fetchSpotPrices(): Promise<[Map<string, Map<string, number>>, string[]]> {
    const output = new Map<string, Map<string, number>>();
    const errors: string[] = [];
    const startTime = new Date(Date.now() - 6 * 3600 * 1000).toISOString();
    const instanceTypes = [...this.specs.keys()].sort();

    await mapConcurrent(this.regions, 4, async (region) => {
      output.set(region, new Map());
      try {
        await mapConcurrent(chunk(instanceTypes, 100), 4, async (part) => {
          const response = await this.aws.runJson(
            [
              "ec2",
              "describe-spot-price-history",
              "--product-descriptions",
              "Linux/UNIX",
              "--start-time",
              startTime,
              "--instance-types",
              ...part,
            ],
            region,
          );
          const newest = new Map<string, { at: number; price: number }>();
          for (const entry of response?.SpotPriceHistory ?? []) {
            const instanceType = String(entry.InstanceType);
            const az = String(entry.AvailabilityZoneId ?? entry.AvailabilityZone ?? "");
            const key = `${instanceType}|${az}`;
            const at = new Date(entry.Timestamp).getTime();
            const price = Number(entry.SpotPrice);
            const current = newest.get(key);
            if (!current || at > current.at) {
              newest.set(key, { at, price });
            }
          }
          const prices = new Map<string, number[]>();
          for (const [key, value] of newest.entries()) {
            const [instanceType] = key.split("|");
            if (!prices.has(instanceType)) prices.set(instanceType, []);
            prices.get(instanceType)!.push(value.price);
          }
          for (const [instanceType, values] of prices.entries()) {
            const minPrice = Math.min(...values);
            const current = output.get(region)!.get(instanceType);
            if (current === undefined || minPrice < current) {
              output.get(region)!.set(instanceType, minPrice);
            }
          }
        });
      } catch (error: any) {
        errors.push(`${region} spot: ${String(error.message ?? error)}`);
      }
    });

    return [output, errors];
  }

  private async fetchSpotScores(): Promise<[Map<string, Map<string, number>>, string[]]> {
    const output = new Map<string, Map<string, number>>();
    const errors: string[] = [];
    for (const region of this.regions) output.set(region, new Map());

    await mapConcurrent([...this.specsByModel.entries()], 4, async ([gpuModel, specs]) => {
      const types = specs.map((spec) => spec.instanceType).sort();
      await mapConcurrent(chunk(this.regions, 10), 3, async (regionGroup) => {
        try {
          const response = await this.aws.runJson(
            [
              "ec2",
              "get-spot-placement-scores",
              "--target-capacity",
              String(this.args.targetInstances),
              "--target-capacity-unit-type",
              "units",
              "--region-names",
              ...regionGroup,
              "--instance-types",
              ...types,
            ],
            this.args.catalogRegion,
          );
          for (const item of response?.SpotPlacementScores ?? []) {
            const region = String(item.Region);
            const score = Number(item.Score);
            output.get(region)?.set(gpuModel, score);
          }
        } catch (error: any) {
          errors.push(`${gpuModel} score: ${String(error.message ?? error)}`);
        }
      });
    });
    return [output, errors];
  }
}

class DashboardApp {
  private readonly screen = blessed.screen({
    smartCSR: true,
    fullUnicode: true,
    title: "Spotter",
  });

  private readonly header = blessed.box({
    parent: this.screen,
    top: 0,
    left: 0,
    width: "100%",
    height: 1,
    tags: true,
    style: { fg: "white", bg: "#102938" },
  });

  private readonly table = blessed.box({
    parent: this.screen,
    top: 1,
    left: 0,
    width: "62%",
    bottom: 12,
    tags: true,
    border: { type: "line" },
    label: " Regions / GPUs ",
    style: { border: { fg: "#3f6e8f" } },
    scrollable: false,
    wrap: false,
  });

  private readonly detail = blessed.box({
    parent: this.screen,
    top: 1,
    left: "62%",
    width: "38%",
    bottom: 12,
    tags: true,
    border: { type: "line" },
    label: " Selection ",
    style: { border: { fg: "#3f6e8f" } },
  });

  private readonly detailSummary = blessed.box({
    parent: this.detail,
    top: 1,
    left: 1,
    width: "100%-2",
    height: 6,
    tags: true,
  });

  private readonly detailCards = blessed.box({
    parent: this.detail,
    top: 7,
    left: 1,
    width: "100%-2",
    bottom: 1,
    tags: true,
  });

  private readonly footer = blessed.box({
    parent: this.screen,
    bottom: 0,
    left: 0,
    width: "100%",
    height: 1,
    tags: true,
    style: { fg: "gray" },
  });

  private readonly loadingOverlay = blessed.box({
    parent: this.screen,
    top: "center",
    left: "center",
    width: 44,
    height: 7,
    tags: true,
    align: "center",
    valign: "middle",
    border: { type: "line" },
    style: {
      fg: "white",
      bg: "#102938",
      border: { fg: "#3f6e8f" },
    },
  });

  private snapshot: RefreshSnapshot = { rows: [], errors: [] };
  private selectedIndex = 0;
  private scrollOffset = 0;
  private refreshErrors: string[] = [];
  private readonly historyCache = new Map<string, HistoryCacheEntry>();
  private refreshTimer?: NodeJS.Timeout;
  private isRefreshing = false;
  private chart: any;
  private chartHeight = 0;
  private chartWidth = 0;
  private lastChartSeries: HistorySeries[] = [];
  private activePane: ActivePane = "regions";
  private detailSelectedIndex = 0;
  private hasLoadedInitialSnapshot = false;

  constructor(private readonly data: TrackerData, private readonly args: Args) {
    this.chart = this.createChart(16);
    this.screen.append(this.chart);
  }

  async run(): Promise<void> {
    this.bindKeys();
    this.renderLoading("Loading AWS GPU catalog...", true);
    this.refreshErrors = await this.data.initialize();
    await this.refresh();
    this.refreshTimer = setInterval(() => {
      void this.refresh();
    }, this.args.interval * 1000);
  }

  destroy(): void {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
    this.screen.destroy();
  }

  private bindKeys(): void {
    this.screen.key(["q", "C-c"], () => {
      this.destroy();
      process.exit(0);
    });
    this.screen.key(["up", "k"], () => this.moveActiveSelection(-1));
    this.screen.key(["down", "j"], () => this.moveActiveSelection(1));
    this.screen.key(["pageup"], () => this.moveActiveSelection(-this.visibleCountForActivePane()));
    this.screen.key(["pagedown"], () => this.moveActiveSelection(this.visibleCountForActivePane()));
    this.screen.key(["home"], () => this.moveActiveToBoundary("start"));
    this.screen.key(["end"], () => this.moveActiveToBoundary("end"));
    this.screen.key(["left"], () => this.setActivePane("regions"));
    this.screen.key(["right"], () => this.setActivePane("selection"));
    this.screen.key(["s"], () => this.cycleSort());
    this.screen.key(["r"], () => {
      void this.refresh(true);
    });
    this.screen.on("resize", () => this.render());
  }

  private cycleSort(): void {
    const selectedKey = this.rowKey(this.snapshot.rows[this.selectedIndex]);
    const current = SORT_MODES.indexOf(this.args.sort);
    this.args.sort = SORT_MODES[(current + 1) % SORT_MODES.length];
    this.snapshot.rows = this.data.sortRows(this.snapshot.rows, this.args.sort);
    if (selectedKey) {
      const nextIndex = this.snapshot.rows.findIndex((row) => this.rowKey(row) === selectedKey);
      this.selectedIndex = nextIndex >= 0 ? nextIndex : 0;
    }
    this.scrollOffset = 0;
    this.render();
    void this.ensureHistory();
  }

  private visibleRowCount(): number {
    return Math.max(6, (Number(this.table.height) || 20) - 4);
  }

  private visibleSelectionRowCount(): number {
    const blockHeight = 7;
    const innerHeight = Math.max(5, (Number(this.detailCards.height) || 10) - 2);
    return Math.max(1, Math.floor(innerHeight / blockHeight));
  }

  private visibleCountForActivePane(): number {
    return this.activePane === "regions" ? this.visibleRowCount() : this.visibleSelectionRowCount();
  }

  private moveActiveSelection(delta: number): void {
    if (this.activePane === "regions") {
      this.moveRegionSelection(delta);
      return;
    }
    this.moveDetailSelection(delta);
  }

  private moveActiveToBoundary(edge: "start" | "end"): void {
    if (this.activePane === "regions") {
      this.setSelection(edge === "start" ? 0 : this.snapshot.rows.length - 1);
      return;
    }
    const optionCount = this.selectedRow()?.options.length ?? 0;
    this.setDetailSelection(edge === "start" ? 0 : optionCount - 1);
  }

  private moveRegionSelection(delta: number): void {
    if (this.snapshot.rows.length === 0) return;
    this.setSelection(Math.max(0, Math.min(this.snapshot.rows.length - 1, this.selectedIndex + delta)));
  }

  private setActivePane(pane: ActivePane): void {
    this.activePane = pane;
    this.render();
  }

  private setSelection(index: number): void {
    if (this.snapshot.rows.length === 0) {
      this.selectedIndex = 0;
      this.scrollOffset = 0;
      this.detailSelectedIndex = 0;
      this.render();
      return;
    }
    this.selectedIndex = Math.max(0, Math.min(index, this.snapshot.rows.length - 1));
    this.detailSelectedIndex = 0;
    const visible = this.visibleRowCount();
    if (this.selectedIndex < this.scrollOffset) {
      this.scrollOffset = this.selectedIndex;
    } else if (this.selectedIndex >= this.scrollOffset + visible) {
      this.scrollOffset = this.selectedIndex - visible + 1;
    }
    this.render();
    void this.ensureHistory();
  }

  private moveDetailSelection(delta: number): void {
    const optionCount = this.selectedRow()?.options.length ?? 0;
    if (optionCount === 0) return;
    this.setDetailSelection(this.detailSelectedIndex + delta);
  }

  private setDetailSelection(index: number): void {
    const optionCount = this.selectedRow()?.options.length ?? 0;
    if (optionCount === 0) {
      this.detailSelectedIndex = 0;
      this.render();
      return;
    }
    this.detailSelectedIndex = Math.max(0, Math.min(optionCount - 1, index));
    this.render();
  }

  private rowKey(row?: ModelRegionRow): string | undefined {
    if (!row) return undefined;
    return `${row.region}|${row.gpuModel}`;
  }

  private selectedRow(): ModelRegionRow | undefined {
    return this.snapshot.rows[this.selectedIndex];
  }

  private async refresh(forceHistory = false, preferredKey?: string): Promise<void> {
    if (this.isRefreshing) return;
    this.isRefreshing = true;
    if (!this.hasLoadedInitialSnapshot) {
      this.renderLoading("Refreshing AWS data...", true);
    } else {
      this.renderLoading("refreshing");
    }
    try {
      const key = preferredKey ?? this.rowKey(this.selectedRow());
      const snapshot = await this.data.refreshSnapshot();
      snapshot.errors = [...this.refreshErrors, ...snapshot.errors];
      this.snapshot = snapshot;
      if (key) {
        const nextIndex = this.snapshot.rows.findIndex((row) => this.rowKey(row) === key);
        this.selectedIndex = nextIndex >= 0 ? nextIndex : 0;
      } else {
        this.selectedIndex = 0;
      }
      this.scrollOffset = 0;
      this.hasLoadedInitialSnapshot = true;
      this.render();
      await this.ensureHistory(forceHistory);
    } catch (error: any) {
      this.snapshot.errors = [...this.refreshErrors, String(error.message ?? error)];
      if (!this.hasLoadedInitialSnapshot) {
        this.renderLoading(`Load failed\n${String(error.message ?? error)}`, true);
      } else {
        this.render();
      }
    } finally {
      this.isRefreshing = false;
    }
  }

  private async ensureHistory(force = false): Promise<void> {
    const row = this.selectedRow();
    if (!row) {
      this.renderChart([]);
      return;
    }
    const key = `${this.rowKey(row)}|${this.args.historyHours}`;
    const current = this.historyCache.get(key);
    if (!force && current && Date.now() - current.fetchedAt < this.args.interval * 1000) {
      this.renderChart(current.series);
      return;
    }
    try {
      const series = await this.data.fetchHistory(row, this.args.historyHours);
      this.historyCache.set(key, { fetchedAt: Date.now(), series });
      if (this.rowKey(this.selectedRow()) === this.rowKey(row)) {
        this.renderChart(series);
      }
    } catch (error: any) {
      this.chart.setLabel(` Spot / GPU Trend (${String(error.message ?? error)}) `);
      this.renderChart([]);
    }
  }

  private renderLoading(message: string, initial = false): void {
    if (initial) {
      this.header.hide();
      this.table.hide();
      this.detail.hide();
      this.chart.hide();
      this.footer.hide();
      this.loadingOverlay.setContent(`{bold}Spotter{/bold}\n\n${message}`);
      this.loadingOverlay.show();
      this.loadingOverlay.setFront();
      this.screen.render();
      return;
    }
    this.loadingOverlay.hide();
    this.header.show();
    this.table.show();
    this.detail.show();
    this.chart.show();
    this.footer.show();
    const screenWidth = Math.max(20, Number(this.screen.width) || 80);
    const leftText = " Spotter";
    const rightText = `${message.toLowerCase()} `;
    const spacing = Math.max(1, screenWidth - stripTags(leftText).length - stripTags(rightText).length);
    this.header.setContent(`{bold}${leftText}{/bold}${" ".repeat(spacing)}${rightText}`);
    this.screen.render();
  }

  private render(): void {
    this.loadingOverlay.hide();
    this.header.show();
    this.table.show();
    this.detail.show();
    this.chart.show();
    this.footer.show();
    const screenHeight = Number(this.screen.height) || 30;
    const bodyTop = 1;
    const footerHeight = 1;
    const availableBodyHeight = Math.max(12, screenHeight - bodyTop - footerHeight);
    const chartHeight = Math.max(8, Math.floor(availableBodyHeight / 3));
    this.syncChartSize(chartHeight);
    this.table.bottom = chartHeight + 1;
    this.detail.bottom = chartHeight + 1;

    const rows = this.snapshot.rows;
    const selected = this.selectedRow();
    const refreshed = this.snapshot.lastRefreshAt
      ? this.snapshot.lastRefreshAt.toLocaleTimeString()
      : "n/a";
    const screenWidth = Math.max(20, Number(this.screen.width) || 80);
    const leftText = " Spotter";
    const rightText = `refreshed ${refreshed} `;
    const spacing = Math.max(1, screenWidth - stripTags(leftText).length - stripTags(rightText).length);
    this.header.setContent(`{bold}${leftText}{/bold}${" ".repeat(spacing)}${rightText}`);

    this.renderTable(rows);
    this.renderDetail(selected);
    this.table.style.border.fg = this.activePane === "regions" ? "cyan" : "#3f6e8f";
    this.detail.style.border.fg = this.activePane === "selection" ? "cyan" : "#3f6e8f";

    if (!selected) {
      this.renderChart([]);
    }

    const errorText =
      this.snapshot.errors.length > 0 ? `   errors: ${this.snapshot.errors.slice(0, 2).join(" | ")}` : "";
    this.footer.setContent(
      ` keys left/right pane  up/down move  s sort  r refresh  q quit${errorText}`,
    );
    this.screen.render();
  }

  private renderTable(rows: ModelRegionRow[]): void {
    const rowLineBudget = Math.max(3, (Number(this.table.height) || 20) - 5);
    const buildWindow = (startIndex: number): ModelRegionRow[] => {
      const window: ModelRegionRow[] = [];
      let linesUsed = 0;
      let previousRegion: string | undefined;
      for (let index = startIndex; index < rows.length; index += 1) {
        const row = rows[index];
        const separatorLines = window.length > 0 && previousRegion !== row.region ? 1 : 0;
        if (linesUsed + separatorLines + 1 > rowLineBudget) break;
        if (separatorLines > 0) linesUsed += separatorLines;
        window.push(row);
        linesUsed += 1;
        previousRegion = row.region;
      }
      return window;
    };

    if (rows.length === 0) {
      this.scrollOffset = 0;
    } else {
      this.scrollOffset = Math.max(0, Math.min(this.scrollOffset, rows.length - 1));
      if (this.selectedIndex < this.scrollOffset) this.scrollOffset = this.selectedIndex;
      let windowed = buildWindow(this.scrollOffset);
      while (
        rows.length > 0 &&
        windowed.length > 0 &&
        this.selectedIndex > this.scrollOffset + windowed.length - 1 &&
        this.scrollOffset < rows.length - 1
      ) {
        this.scrollOffset += 1;
        windowed = buildWindow(this.scrollOffset);
      }
    }

    const windowed = buildWindow(this.scrollOffset);
    const rawTableWidth =
      typeof this.table.width === "number"
        ? this.table.width
        : Math.floor((Number(this.screen.width) || 120) * 0.62);
    const innerWidth = Math.max(40, rawTableWidth - 3);
    const regionWidth = 15;
    const gpuWidth = 6;
    const scoreWidth = 5;
    const azWidth = 5;
    const vramWidth = 9;
    const spotWidth = 8;
    const odWidth = 8;
    const scoreAzGap = 2;
    const availVramGap = 4;
    const vramSpotGap = 3;
    const fixedWidth =
      5 +
      regionWidth +
      1 +
      gpuWidth +
      1 +
      scoreWidth +
      scoreAzGap +
      azWidth +
      1 +
      10 +
      availVramGap +
      vramWidth +
      vramSpotGap +
      spotWidth +
      1 +
      odWidth;
    const header =
      `${fitCell("#", 4)} ` +
      `${fitCell("Region", regionWidth)} ` +
      `${fitCell("GPU", gpuWidth)} ` +
      `${fitCell("Score", scoreWidth)}` +
      `${" ".repeat(scoreAzGap)}` +
      `${fitCell("AZ", azWidth)} ` +
      `${fitCell("Avail", 10)}` +
      `${" ".repeat(availVramGap)}` +
      `${fitCell("VRAM (GB)", vramWidth)}` +
      `${" ".repeat(vramSpotGap)}` +
      `${fitCell("Spot/GPU", spotWidth)} ` +
      `${fitCell("OD/GPU", odWidth)}`;
    const lines = [`{bold}${header}{/bold}`, `{gray-fg}${"─".repeat(Math.max(20, stripTags(header).length))}{/gray-fg}`];
    windowed.forEach((row, localIndex) => {
      const actualIndex = this.scrollOffset + localIndex;
      if (localIndex > 0 && windowed[localIndex - 1]?.region !== row.region) {
        lines.push(`{gray-fg}${"─".repeat(Math.max(20, innerWidth))}{/gray-fg}`);
      }
      const pointer = actualIndex === this.selectedIndex ? "›" : " ";
      const scoreText = row.spotScore === undefined ? "n/a" : `${String(row.spotScore).padStart(2, " ")}/10`;
      const azText = formatAzCoverage(row.azCount, row.totalAzCount);
      const color = scoreColor(row.spotScore);
      const content =
        `${pointer} ${String(actualIndex + 1).padStart(2, " ")} ` +
        `${fitCell(row.region, regionWidth)} ` +
        `${fitCell(row.gpuModel, gpuWidth)} ` +
        `{${color}-fg}${fitCell(scoreText, scoreWidth)}{/${color}-fg} ` +
        `${" ".repeat(scoreAzGap)}` +
        `${fitCell(azText, azWidth)} ` +
        `{${color}-fg}${scoreBar(row.spotScore, 10)}{/${color}-fg}` +
        `${" ".repeat(availVramGap)}` +
        `${fitCell(String(Math.round(row.gpuMemoryGiB)), vramWidth)}` +
        `${" ".repeat(vramSpotGap)}` +
        `${fitCell(formatMoney(row.bestSpotGpu).padStart(spotWidth, " "), spotWidth)} ` +
        `${fitCell(formatMoney(row.bestOnDemandGpu).padStart(odWidth, " "), odWidth)}`;
      if (actualIndex === this.selectedIndex) {
        lines.push(`{white-fg}{blue-bg}${content}{/blue-bg}{/white-fg}`);
      } else {
        lines.push(content);
      }
    });
    if (rows.length === 0) {
      lines.push("No rows matched the current filters.");
    } else {
      const visibleEnd = this.scrollOffset + windowed.length;
      lines.push(
        `{gray-fg}showing ${this.scrollOffset + 1}-${Math.min(visibleEnd, rows.length)} of ${rows.length}{/gray-fg}`,
      );
    }
    this.table.setContent(lines.join("\n"));
  }

  private renderDetail(row?: ModelRegionRow): void {
    if (!row) {
      this.detailSummary.setContent("No selection");
      this.detailCards.setContent("");
      return;
    }
    const summaryWidth = Math.max(20, (Number(this.detail.width) || 40) - 4);
    const summaryGapWidth = 2;
    const summaryLeftWidth = Math.max(12, Math.floor((summaryWidth - summaryGapWidth) / 2));
    const summaryRightWidth = Math.max(12, summaryWidth - summaryLeftWidth - summaryGapWidth);
    const summaryRow = (leftLabel: string, leftValue: string, rightLabel: string, rightValue: string): string => {
      const left = `${leftLabel}: ${leftValue}`;
      const right = `${rightLabel}: ${rightValue}`;
      return `${fitTaggedCell(left, summaryLeftWidth)}${" ".repeat(summaryGapWidth)}${fitTaggedCell(right, summaryRightWidth)}`;
    };
    this.detailSummary.setContent(
      [
        `{bold}${row.region} / ${row.gpuModel}{/bold}`,
        summaryRow(
          "Score",
          `{${scoreColor(row.spotScore)}-fg}${row.spotScore ?? "n/a"}{/${scoreColor(row.spotScore)}-fg}/10`,
          "AZ",
          formatAzCoverage(row.azCount, row.totalAzCount),
        ),
        summaryRow("Types", String(row.typeCount), "GPUs", compactRange(row.gpuCountRange)),
        summaryRow(
          "VRAM",
          `${Math.round(row.gpuMemoryGiB)}${mutedUnit("gb")}`,
          "Card",
          `${this.detailSelectedIndex + 1}/${row.options.length}`,
        ),
        summaryRow(
          "Spot",
          `${formatMoney(row.bestSpotInstance)}${mutedUnit("/h")}`,
          "OD",
          `${formatMoney(row.bestOnDemandInstance)}${mutedUnit("/h")}`,
        ),
      ].join("\n"),
    );
    this.detailSelectedIndex = Math.max(0, Math.min(this.detailSelectedIndex, row.options.length - 1));
    const visibleBlocks = this.visibleSelectionRowCount();
    const startIndex = Math.max(
      0,
      Math.min(
        this.detailSelectedIndex - Math.floor(visibleBlocks / 2),
        Math.max(0, row.options.length - visibleBlocks),
      ),
    );
    const visibleOptions = row.options.slice(startIndex, startIndex + visibleBlocks);
    const innerWidth = Math.max(20, (Number(this.detail.width) || 40) - 6);
    const contentWidth = innerWidth - 2;
    const gapWidth = 2;
    const leftColWidth = Math.max(12, Math.floor((contentWidth - gapWidth) / 2));
    const rightColWidth = Math.max(12, contentWidth - leftColWidth - gapWidth);
    const lines: string[] = [];

    const metricRow = (leftLabel: string, leftValue: string, rightLabel: string, rightValue: string): string => {
      const left = `${leftLabel}: ${leftValue}`;
      const right = `${rightLabel}: ${rightValue}`;
      return `│${fitTaggedCell(left, leftColWidth)}${" ".repeat(gapWidth)}${fitTaggedCell(right, rightColWidth)}│`;
    };

    visibleOptions.forEach((option, localIndex) => {
      const optionIndex = startIndex + localIndex;
      const selectedOption = optionIndex === this.detailSelectedIndex;
      const totalVram = Math.round(option.gpuMemoryGiB * option.gpuCount);
      const topBorder = `┌${"─".repeat(Math.max(4, innerWidth - 2))}┐`;
      const bottomBorder = `└${"─".repeat(Math.max(4, innerWidth - 2))}┘`;
      const body = [
        `│${fitCell(`${selectedOption ? "› " : "  "}${option.instanceType}`, contentWidth)}│`,
        metricRow(
          "GPUs",
          `${option.gpuCount} x ${Math.round(option.gpuMemoryGiB)}${mutedUnit("gb")}`,
          "VRAM",
          `${totalVram}${mutedUnit("gb")}`,
        ),
        metricRow("CPU", String(option.vcpus), "RAM", `${Math.round(option.memoryGiB)}${mutedUnit("gb")}`),
        metricRow(
          "Spot",
          `${formatMoney(option.spotInstance)}${mutedUnit("/h")}`,
          "OD",
          `${formatMoney(option.onDemandInstance)}${mutedUnit("/h")}`,
        ),
        metricRow("AZs", String(option.azCount), "Net", compactNetwork(option.networkPerformance)),
      ];

      if (selectedOption) {
        lines.push(`{white-fg}${topBorder}{/white-fg}`);
        lines.push(`{bold}{white-fg}${body[0]}{/white-fg}{/bold}`);
        lines.push(`{white-fg}${body[1]}{/white-fg}`);
        lines.push(`{white-fg}${body[2]}{/white-fg}`);
        lines.push(`{white-fg}${body[3]}{/white-fg}`);
        lines.push(`{cyan-fg}${body[4]}{/cyan-fg}`);
        lines.push(`{white-fg}${bottomBorder}{/white-fg}`);
      } else {
        lines.push(`{gray-fg}${topBorder}{/gray-fg}`);
        lines.push(`{bold}${body[0]}{/bold}`);
        lines.push(body[1]);
        lines.push(body[2]);
        lines.push(body[3]);
        lines.push(`{gray-fg}${body[4]}{/gray-fg}`);
        lines.push(`{gray-fg}${bottomBorder}{/gray-fg}`);
      }
    });

    if (row.options.length === 0) {
      lines.push("No instance options.");
    } else {
      lines.push(
        `{gray-fg}showing ${startIndex + 1}-${Math.min(startIndex + visibleBlocks, row.options.length)} of ${row.options.length}{/gray-fg}`,
      );
    }
    this.detailCards.setContent(lines.join("\n"));
  }

  private renderChart(series: HistorySeries[]): void {
    this.lastChartSeries = series;
    this.chart.setLabel(` Spot / GPU Trend (${this.args.historyHours}h) `);
    if (series.length === 0) {
      this.chart.options.showLegend = false;
      this.chart.options.showNthLabel = 1;
      this.chart.options.minY = 0;
      this.chart.options.maxY = 1;
      this.chart.setData([{ title: "", x: ["", ""], y: [0, 0], style: { line: "black" } }]);
      this.screen.render();
      return;
    }
    const timestamps = [...new Set(series.flatMap((entry) => entry.points.map((point) => point.at.toISOString())))].sort();
    if (timestamps.length === 0) {
      this.chart.options.showLegend = false;
      this.chart.options.showNthLabel = 1;
      this.chart.options.minY = 0;
      this.chart.options.maxY = 1;
      this.chart.setData([{ title: "", x: ["", ""], y: [0, 0], style: { line: "black" } }]);
      this.screen.render();
      return;
    }
    const colors = ["cyan", "yellow", "green", "magenta"] as const;
    const shownSeries = series.slice(0, 4).map((entry, index) => {
      const byTimestamp = new Map(entry.points.map((point) => [point.at.toISOString(), point.value]));
      const firstKnown = entry.points[0]?.value ?? 0;
      let lastValue = firstKnown;
      const values = timestamps.map((stamp) => {
        const current = byTimestamp.get(stamp);
        if (current !== undefined) {
          lastValue = current;
          return current;
        }
        return lastValue;
      });
      return {
        title: entry.title,
        color: colors[index % colors.length],
        values,
      };
    });
    const labelStep = Math.max(1, Math.ceil(timestamps.length / 8));
    const xLabels = timestamps.map((stamp) => formatHourLabel(new Date(stamp)));
    const values = shownSeries.flatMap((entry) => entry.values);
    const maxValue = Math.max(...values);
    const paddedRange = Math.max(maxValue, 1) * 1.1;
    const targetStep = paddedRange / Math.max(1, this.chart.options.numYLabels - 1);
    const step = niceStep(targetStep);
    const minY = 0;
    const maxY = Math.ceil((maxValue + step * 0.25) / step) * step;

    this.chart.options.showLegend = true;
    this.chart.options.showNthLabel = labelStep;
    this.chart.options.minY = 0;
    this.chart.options.maxY = Number(Math.max(maxY, step).toFixed(3));
    this.chart.setData(
      shownSeries.map((entry) => ({
        title: entry.title,
        x: xLabels,
        y: entry.values.map((value) => Number(value.toFixed(3))),
        style: { line: entry.color },
      })),
    );
    this.screen.render();
  }

  private createChart(height: number): any {
    return contrib.line({
      screen: this.screen,
      left: 0,
      bottom: 1,
      width: "100%",
      height,
      border: { type: "line", fg: "#3f6e8f" },
      label: " Spot / GPU Trend ",
      showLegend: true,
      legend: { width: 18 },
      wholeNumbersOnly: false,
      minY: 0,
      style: {
        text: "white",
        baseline: "white",
        line: "cyan",
        border: { fg: "#3f6e8f" },
      },
      xLabelPadding: 6,
      xPadding: 8,
      numYLabels: 6,
    });
  }

  private syncChartSize(height: number): void {
    const nextWidth = Number(this.screen.width) || 0;
    if (this.chartHeight === height && this.chartWidth === nextWidth) {
      return;
    }

    this.chartHeight = height;
    this.chartWidth = nextWidth;
    const previousChart = this.chart;
    this.chart = this.createChart(height);
    this.screen.append(this.chart);
    if (previousChart) {
      previousChart.detach();
    }
    if (this.lastChartSeries.length > 0) {
      this.renderChart(this.lastChartSeries);
    }
  }
}

function renderOnce(snapshot: RefreshSnapshot): void {
  const lines = [
    "Spotter",
    `refreshed ${snapshot.lastRefreshAt?.toISOString() ?? "n/a"} | rows ${snapshot.rows.length}`,
    "",
  ];
  for (const [index, row] of snapshot.rows.slice(0, 20).entries()) {
    lines.push(
      `${String(index + 1).padStart(2, " ")} ${row.region.padEnd(12)} ${row.gpuModel.padEnd(6)} score ${String(
        row.spotScore ?? "n/a",
      ).padStart(3)} az ${formatAzCoverage(row.azCount, row.totalAzCount).padStart(5)} spot/gpu ${formatMoney(row.bestSpotGpu).padStart(8)} od/gpu ${formatMoney(
        row.bestOnDemandGpu,
      ).padStart(8)} best ${row.bestType}`,
    );
  }
  if (snapshot.errors.length > 0) {
    lines.push("");
    lines.push(`Errors: ${snapshot.errors.join(" | ")}`);
  }
  process.stdout.write(`${lines.join("\n")}\n`);
}

async function main(): Promise<number> {
  const args = parseArgs(process.argv.slice(2));
  const data = new TrackerData(args);

  if (args.once || !process.stdout.isTTY) {
    const refreshErrors = await data.initialize();
    const snapshot = await data.refreshSnapshot();
    snapshot.errors = [...refreshErrors, ...snapshot.errors];
    renderOnce(snapshot);
    return 0;
  }

  const app = new DashboardApp(data, args);
  await app.run();
  return 0;
}

main().catch((error: unknown) => {
  process.stderr.write(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
