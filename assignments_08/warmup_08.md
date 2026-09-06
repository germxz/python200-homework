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

You need to store 10 TB of image files and retrieve them by filename from any machine. A: Object storage - AWS S3

You need to run an ML training job on a GPU for four hours, then shut it down. A: Compute (GPU compute) - AWS EC2 p3.2xlarge

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets. A: Serverless compute - AWS Lambda

You need to send structured data to a large language model and get a text response back. A: LLM API - AWS Bedrock

# Cloud Concepts Question 4

IaaS is the most primitive form of service — it provides raw virtualized infrastructure (compute, storage, and networking) that you configure and manage yourself. An example is AWS EC2. I am responsible for managing the operating system, runtime, middleware, and security patching; the provider only manages the physical hardware and virtualization layer.

PaaS takes it down a level and manages a bit more for a developer, saving them time so they can focus on building. An example is Microsoft Azure's App Service, which handles scalability, environments, and the OS for me. I am responsible for my application code and data; the provider manages the OS, runtime, and scaling infrastructure underneath it.

SaaS is a complete, ready-to-use application delivered over the internet, with everything managed by the provider. An example is Gmail. I am responsible for essentially nothing beyond how I use the software and my own data within it; the provider manages the entire stack — infrastructure, platform, application, updates, and maintenance.

# Managed Data Platforms (Databricks / Snowflake)

A managed data platform allows users to use preconfigured infrastructures built on top of infrastructure services like AWS that allow for a much easier experience working with large-scale data management. You would be able to use analytical functions, scale a really big database, or optimize it with the tools provided. The cost is money and less custom configurations available to you compared to setting everything up yourself on AWS directly.

# Cloud Concepts Q5

A situation where you probably won't need the cloud is if you already have the sophisticated hardware to host an application. The cloud provides elasticity but if traffic isn't expected to spike, there will be no need. The tradeoff here is needing to provide maintenance on the hardware like security and patches.

## Part 2

# Cloud Landscape Q1

Amazon Web Service
AWS's primary strength is having the broadest, most mature catalog of services and configurations of any provider, making it the go-to for startups and enterprises that want maximum flexibility without being locked into a narrow toolset.

Google Cloud Platform
GCP's primary strength is machine learning and large-scale data analytics, built on the same technology (like TensorFlow and BigQuery) Google uses internally, making it a natural fit for data-and-ML-heavy companies like Spotify, which uses it to build recommendation algorithms based on listening history.

Microsoft Azure
Azure's primary strength is its deep integration with the Microsoft ecosystem (Windows Server, Active Directory, Office 365), making it especially competitive with enterprises and government agencies already standardized on Microsoft tools.

# Cloud Landscape Q2

One reason why the course switched from Azure to supabase was because it is easier to access. Azure needs users to join an organization and verification and it would affect access to late students. Supabase allows students to do the job without having to worry about authentication measures.

Another reason why the course switched was because the Azure setup stored data as flat files rather than in a structured database, whereas Supabase provides a Postgres database with rows and columns by default. This makes it usable for cleaning and other data operations right out the box.

The third reason why the course switched was because Supabase provides tools needed to compare two databases in the coming assignments. This will make the pipeline database easier to inspect and debug.

# Cloud Landscape Q3

You need to store 10 TB of image files and retrieve them by filename from any machine. A: Object storage - AWS S3

You need to run an ML training job on a GPU for four hours, then shut it down. A: Compute (GPU compute) - AWS EC2 p3.2xlarge

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets. A: Serverless compute - AWS Lambda

You need to send structured data to a large language model and get a text response back. A: LLM API - AWS Bedrock

# Cloud Landscape Q4

A simple data project could be taking data from all recorded soccer games in the world and displaying different stats for the everchanging data. There are thousands of soccer games going on in the world everyday. I could use a managed relational database because soccer data will usually be structured by player, score, teams, and other metadata. Then I would use serverless compute to run the backend since it does not need a server 24/7 and will scale to fit server needs.

The benefit from consolidation to one platform would be more ease of access and compatibility of tools. Since these platforms have overlapping functionality, you can often cover most of what you need without leaving the platform, so you're not sacrificing much capability for the convenience gained. You will give up having to pay one bill for the job as using different platforms means getting charged on more than one.
