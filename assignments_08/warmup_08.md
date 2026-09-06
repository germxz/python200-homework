## Part 1

# Cloud Concepts Q1

The core economic model of cloud computing is that people will rent from these companies that own cloud servers and they pay a bill like it were a utility. It's different from owning servers because you don't need to own the hardware required for hosting servers.

# Cloud Concepts Q2

Vertical scaling practically means uprading the hardware rather than acquiring more units for computing. Acquiring more units is horizontal scaling because it would take up more space in a room. I will choose vertical scaling when I have minimal work to do that doesn't require much computing power and the task is done in par time. Horizontal Scaling would be good for loads that are heavy, need constant uptime, and compute bigger loads than what a single unit can do.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
A: Horizontal scaling.

A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
A: Vertical Scaling

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
A: Horizontal Scaling

# Cloud Concepts Q3

## Part A — Classification of each item (IaaS / PaaS / SaaS / BaaS)

1. **Gmail — SaaS.** Google manages the hardware, the platform, and the application itself, so the software is ready to use with no setup or maintenance on my end.

2. **Azure Virtual Machines — IaaS.** Azure gives me a virtual machine and I configure it myself, installing the OS, runtime, and everything my application needs on top of it.

3. **AWS S3 (Simple Storage Service) — IaaS.** S3 is raw storage infrastructure. AWS manages the physical hardware and durability, but I decide how the data is structured, accessed, and secured. It is not BaaS, because BaaS means a bundled set of backend services packaged together for app developers, while S3 is a single, general-purpose infrastructure building block I would have to combine with other services myself.

4. **GitHub Codespaces — PaaS.** It gives me a ready-to-code cloud development environment (a container with my repo, dependencies, and VS Code in the browser) without provisioning or managing any servers. I am building on a platform, not configuring infrastructure.

5. **Snowflake — PaaS.** It is a managed data platform I build queries, pipelines, and analytics on top of, with all the underlying infrastructure, scaling, and maintenance handled for me — closer to a development platform than a finished, ready-to-use application.

6. **Supabase — BaaS.** Supabase bundles a managed Postgres database, authentication, storage, and serverless functions together as ready-made backend services, so I build my own frontend on top of an already-assembled backend instead of assembling one myself. That bundling is what makes it BaaS rather than IaaS (a single raw building block like S3) or SaaS (a finished application like Gmail).

## Part B — IaaS, PaaS, and SaaS described

**IaaS** is the most primitive form of service — it provides raw virtualized infrastructure (compute, storage, and networking) that you configure and manage yourself. Example: AWS EC2. The developer manages the operating system, runtime, middleware, the application, and security patching; the provider manages only the physical hardware, networking, and virtualization layer.

**PaaS** takes it down a level and manages more for the developer, which saves time so they can focus on building. Example: Microsoft Azure App Service. The developer manages only their application code and data; the provider manages the operating system, runtime, middleware, scaling, and all the underlying infrastructure.

**SaaS** is a complete, ready-to-use application delivered over the internet. Example: Gmail. The developer or user manages nothing but their own data and how they use the software; the provider manages the entire stack — infrastructure, platform, application, updates, and maintenance.

# Cloud Concepts Question 4

A managed data platform allows users to use preconfigured infrastructures built on top of infrastructure services like AWS that allow for a much easier experience working with large-scale data management. You would be able to use analytical functions, scale a really big database, or optimize it with the tools provided. The cost is money and less custom configurations available to you compared to setting everything up yourself on AWS directly.

# Cloud Concepts Q5

A situation where you probably won't need the cloud is if you already have the sophisticated hardware to host an application. The cloud provides elasticity but if traffic isn't expected to spike, there will be no need. The tradeoff here is needing to provide maintenance on the hardware like security and patches.

## Part 2

# Cloud Landscape Q1

The three hyperscalers are Amazon Web Services (AWS), Google Cloud Platform (GCP), and Microsoft Azure.

**Hyperscaler 1: Amazon Web Services (AWS)**
AWS's primary strength is having the broadest, most mature catalog of services and configurations of any provider. The organizations most likely to use it are startups and enterprises that want maximum flexibility without being locked into a narrow toolset.

**Hyperscaler 2: Google Cloud Platform (GCP)**
GCP's primary strength is machine learning and large-scale data analytics, built on the same technology (like TensorFlow and BigQuery) Google uses internally. The organizations most likely to use it are data- and ML-heavy companies such as Spotify, which uses it to build recommendation algorithms based on listening history.

**Hyperscaler 3: Microsoft Azure**
Azure's primary strength is its deep integration with the Microsoft ecosystem (Windows Server, Active Directory, Office 365). The organizations most likely to use it are enterprises and government agencies already standardized on Microsoft tools.

# Cloud Landscape Q2

One reason why the course switched from Azure to supabase was because it is easier to access. Azure needs users to join an organization and verification and it would affect access to late students. Supabase allows students to do the job without having to worry about authentication measures.

Another reason why the course switched was because the Azure setup stored data as flat files rather than in a structured database, whereas Supabase provides a Postgres database with rows and columns by default. This makes it usable for cleaning and other data operations right out the box.

The third reason why the course switched was because Supabase provides tools needed to compare two databases in the coming assignments. This will make the pipeline database easier to inspect and debug.

# Cloud Landscape Q3

You need to store 10 TB of image files and retrieve them by filename from any machine.
A: Object storage — AWS S3

You need to run an ML training job on a GPU for four hours, then shut it down.
A: Compute — AWS EC2 (p3.2xlarge GPU instance)

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
A: Serverless compute — AWS Lambda

You need to send structured data to a large language model and get a text response back.
A: LLM API — AWS Bedrock

# Cloud Landscape Q4

A simple data project could be taking data from recorded soccer games around the world and displaying different stats for that constantly changing data. There are thousands of soccer games going on in the world every day.

My stack would combine two different providers from the taxonomy table:

1. **Managed relational DB — Supabase (Postgres).** Soccer data is naturally structured by player, team, score, and match metadata, so a relational database with rows and columns fits it well, and Supabase gives me a Postgres database without managing a server.

2. **Serverless compute — AWS Lambda.** Lambda runs the backend API that queries the database and returns stats. It does not need a server running 24/7 and scales up automatically when traffic spikes, such as during a big match, then scales back down when it quiets.

The benefit of consolidating to one platform instead would be more ease of access and compatibility of tools. Since these platforms have overlapping functionality, you can often cover most of what you need without leaving the platform, so you're not sacrificing much capability for the convenience gained. What you give up by splitting across two providers is the simplicity of one bill and one dashboard, since using different platforms means getting charged and managing credentials in more than one place.
