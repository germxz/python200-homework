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

Gmail
SaaS. Google manages handling of data and maintenance and the sofware is ready to use.

Azure Virtual Machines
IaaS. Infrastructure that lets you interface with your machine more directly leading to more user configuration while you pay for the tools.

AWS S3 (Simple Storage Service)
IaaS. A basic building block you assemble into whatever you're building, rather than a ready-made application or a platform for building one.

GitHub Codespaces
PaaS. It gives you a ready-to-code cloud development environment (a container with your repo, dependencies, and VS Code in the browser) without you provisioning or managing any servers yourself. You're building on a platform, not configuring infrastructure.

Snowflake
SaaS. You log in and start running queries; Snowflake handles all the infrastructure, scaling, patching, and maintenance behind the scenes. It's marketed and consumed like a finished product (a "data platform" you use).

Supabase
BaaS. It gives you a managed Postgres database, authentication, storage, and serverless functions as building blocks for an app you're developing, rather than a finished end-user application or raw infrastructure.

IaaS is the most primitive form of service as it interacts more intimately with your machine. This means you are given the tools to configure and manage more; which will take more work. An example would be using AWS to tailor your server, storage and networking. My job would be bulding the pipelines needed to configure my application to perfection

PaaS takes it down a level and manages a bit more for a developer and it saves them time so they can focus on building and such. An example I can think of is using the tools that Microsoft Azure provides to handle scalability, environments, and the OS for me without having to create my own tools. My job would be to focus on building and deploying quickly and efficiently if I were using a PaaS.

SaaS is an application with a set of tools that allows you to complete a set of tasks. An example of it would be a service that is ready to use hassle-free in terms of programming. The job here would be to simply use the service and be worry free of anything as I have all the tools I paid for.

# Cloud Concepts Question 4

A managed data platform allows users to use preconfigured infrastructures built on top of infrastructure services like AWS that allow for a much easier experience working with huge database management. You would be able to use analytical functions, scale a really big database optimize it with the tools provided. The cost is money and less custom configurations available to you.

# Cloud Concepts Q5

A situation where you probably won't need the cloud is if you already have the sophisticated hardware to host an application. The cloud provides elasticity but if traffic isn't expected to spike, there will be no need. The tradeoff here is needing to provide maintenance on the hardware like security and patches.

## Part 2

# Cloud Landscape Q1

Amazon Web Service
AWS is the oldest and largest and it is seen in 33% of the cloud market. It also has a widest slection of configurations so it could be used by start-ups.

Google Cloud Platform
GCP specializes in Machine learning and data. It's good for large-scale analytics or for machine learning. Spotify would use this to build machine learning algorithms based on a user's listening record and predict the next best song.

Microsoft Azure
It is the competitive in enterprise and government settings because of its integration to windows. Most if not all government operating systems are conquered by Microsoft and Azure provides a wide range of services from databases to computing power.

# Cloud Landscape Q2

One reason why the course switched from Azure to supabase was because it is easier to access. Azure needs users to join an organization and verification and it would affect access to late students. Supabase allows students to do the job without having to worry about authentication measures.

Another reason why the course switched was because the Azure setup stored data as flat files rather than in a structured database, whereas Supabase provides a Postgres database with rows and columns by default. This makes it usable for cleaning and other data operations right out the box.

The third reason why the course switched was because Supabase provides tools needed to compare two databases in the coming assignments. This will make the pipeline database easier to inspect and debug.

# Cloud Landscape Q3

You need to store 10 TB of image files and retrieve them by filename from any machine.
A: Object storage - AWS S3

You need to run an ML training job on a GPU for four hours, then shut it down.
A: ML Platform - SageMaker AWS

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
A: Serverless compute - AWS Lambda

You need to send structured data to a large language model and get a text response back.
A: LLM API - AWS Bedrock

# Cloud Landscape Q4

A simple data project could be taking data from all recorded soccer games in the world and displaying different stats for the everchanging data. There are thousands of soccer games going on in the world everyday. I could use a managed relational database because soccer data will usually be structured by player, score, teams, and other metadata. Then I would use serverless compute to run the backend since it does not need a server 24/7 and will scale to fit server needs.

The benefit from consolidation to one platform would be more ease of access and compatibility of tools. Since these platforms have overlapping functionality, you can often cover most of what you need without leaving the platform, so you're not sacrificing much capability for the convenience gained. You will give up having to pay one bill for the job as using different platforms means getting charged on more than one.
