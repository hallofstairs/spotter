# spotter

Terminal dashboard for choosing AWS GPU instances for training jobs.

It shows, by region and GPU family:
- inferred availability signals
- Spot placement score
- Spot and On-Demand pricing
- recent Spot price trend

Defaults are focused on training-relevant NVIDIA GPUs like `H100`, `H200`, `A100`, `L40S`, `L4`, `A10G`, and `V100`.

## Requirements

- Node.js 20+
- AWS CLI v2
- AWS credentials configured locally

Suggested IAM permissions:
- `ec2:DescribeRegions`
- `ec2:DescribeAvailabilityZones`
- `ec2:DescribeInstanceTypes`
- `ec2:DescribeInstanceTypeOfferings`
- `ec2:DescribeSpotPriceHistory`
- `ec2:GetSpotPlacementScores`
- `pricing:GetProducts`

## Install

From a local clone:

```bash
npm install
npm link
spotter
```

After publish:

```bash
npm install -g spotter
spotter
```

Directly from a git repo:

```bash
npm install -g <git-url>
spotter
```

## Usage

Run the dashboard:

```bash
spotter
```

One-shot snapshot:

```bash
spotter --once
```

Useful examples:

```bash
spotter --regions us-east-1,us-west-2 --gpu-models H100,A100
spotter --regions all
spotter --target-instances 8
spotter --history-hours 24
spotter --profile my-profile
```


## Notes

- `npm link` exposes the CLI globally as `spotter`.
- `Score`: EC2 Spot placement score from `1-10`; higher is better
- `AZ`: `offered/total` Availability Zone coverage for that GPU family in the region


