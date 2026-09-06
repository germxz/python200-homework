## Part 1

# Cloud Concepts Q1

The core economic model of cloud computing is that people will rent from these companies that own cloud servers and they pay a bill like it were a utility. It's different from owning servers because you don't need to buy the hardware upfront or keep it running, and you can stop paying once you stop using it.

# Cloud Concepts Q2

Vertical scaling practically means uprading the hardware rather than acquiring more units for computing. Acquiring more units is horizontal scaling because it would take up more space in a room. I will choose vertical scaling when I have minimal work to do that doesn't require much computing power and the task is done in par time. Horizontal Scaling would be good for loads that are heavy, need constant uptime, and compute bigger loads than what a single unit can do.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
A: Horizontal scaling. Web requests don't depend on each other so you can just add more servers to take the load instead of making one server bigger.

A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
A: Vertical Scaling. The job is running on one machine so the only fix is upgrading that machine's hardware.

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
A: Horizontal Scaling. The files can be handled separately so the work splits across a bunch of machines running at the same time.

# Cloud Concepts Q3

Gmail
SaaS. It is an application someone else built, runs, and maintains. I just log in and use it.

Azure Virtual Machines
IaaS. Infrastructure that lets you interface with your machine more directly leading to more user configuration while you pay for the tools.

AWS S3 (Simple Storage Service)
IaaS. S3 is one single infrastructure building block, just raw storage you access by key. I would still have to bring in separate compute, database and authentication services to build anything complete, which is what makes it different from a BaaS bundle that comes already wired together.

GitHub Codespaces
PaaS. The provider handles all the infrastructure and I bring my own code. It gives me a ready to code environment in the browser without setting up any servers.

Snowflake
PaaS. It is a managed platform, not a raw building block and not a finished app. Snowflake runs the infrastructure, scaling and maintenance and I bring my own queries and pipelines to run on it.

Supabase
BaaS. Supabase sits higher up than raw infrastructure and hands me application level pieces like a Postgres database, authentication, file storage and functions, all wired together already and reachable through an API. I build my frontend on a backend that is already put together instead of assembling one myself out of parts like S3.

IaaS is the most primitive form of service as it interacts more intimately with your machine. This means you are given the tools to configure and manage more, which will take more work. An example is AWS EC2. My job would be managing the operating system, the runtime, my application and the security patching, while the provider only handles the physical hardware and virtualization.

PaaS takes it down a level and manages a bit more for a developer and it saves them time so they can focus on building. An example is GitHub Codespaces. My job would be managing only my code and my data, while the provider handles the OS, the runtime, the scaling and everything underneath.

SaaS is an application with a set of tools that allows you to complete a set of tasks. An example is Gmail. My job here is nothing besides my own data and how I use the software, since the provider manages the whole stack including updates and maintenance.

# Cloud Concepts Q4

A managed data platform like Databricks or Snowflake is a preconfigured data environment built on top of infrastructure from a provider like AWS or GCP, so you never touch that infrastructure yourself.

What you gain is convenience. The platform already handles the provisioning, the cluster management, the scaling and the maintenance, and it comes with analytical tools ready to go. That means a small team can start querying and building pipelines right away instead of putting that whole stack together themselves on AWS or GCP.

What you give up is control and money. You can only configure what the platform lets you configure, so a weird or highly custom workload might not fit how it works. You also pay a premium on top of the cloud costs it is already passing through, and your pipelines end up tied to that platform's way of doing things which makes leaving harder later.

# Cloud Concepts Q5

The situation is when your dataset fits comfortably on a single machine and you don't have massive compute demands. Local processing is usually faster and cheaper there since you skip renting infrastructure and skip the time spent uploading data and setting services up. This is especially true early on when you are still prototyping with a small dataset.

## Part 2

# Cloud Landscape Q1

The three hyperscalers are Amazon Web Services, Google Cloud Platform and Microsoft Azure.

Amazon Web Services
AWS has the widest and most mature selection of services and configurations out of any provider, so it fits startups and enterprises that want maximum flexibility without getting locked into a narrow set of tools.

Google Cloud Platform
GCP specializes in machine learning and large scale data analytics built on the same technology Google uses internally, so it fits data and ML heavy companies like Spotify that build recommendation algorithms off listening history.

Microsoft Azure
Azure integrates deeply with the Microsoft ecosystem like Windows Server, Active Directory and Office 365, so it fits enterprises and government agencies that already run on Microsoft tools.

# Cloud Landscape Q2

One reason the course switched from Azure to Supabase is access. Azure requires organizational provisioning which delays students getting in, while a Supabase account is self provisioned at supabase.com in under two minutes.

Another reason is how well it fits what we are learning. Supabase is a relational database you query with SQL, and querying, filtering and reasoning about structured data is a skill that carries over into almost every data role instead of being specific to one platform.

The third reason is pipeline coherence. The raw and enriched zones of the ETL pipeline map cleanly onto two tables with a clear relationship between them, which makes every stage easy to inspect and debug.

My reflection on this is that picking a cloud tool for a new project is less about which provider is the best overall and more about fit. What matters is what my team already uses, where the data and the expertise already live, whether I need a specific service that has no equivalent somewhere else, and what the pricing looks like for how I would actually use it. How fast you can get started and how transferable the skill is can matter more than raw capability, which is exactly why Supabase beat Azure here even though Azure is the much bigger platform.

# Cloud Landscape Q3

You need to store 10 TB of image files and retrieve them by filename from any machine.
A: Object storage. AWS S3. You are storing files and pulling them back by key, which is exactly what object storage does, and it is reachable from any machine over the internet.

You need to run an ML training job on a GPU for four hours, then shut it down.
A: Compute. AWS EC2, using a GPU instance like the p3.2xlarge. You are spinning up a server, running your code on it, then tearing it down, and you only pay for the hours it was up.

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
A: Serverless compute. AWS Lambda. There is no server for me to manage and it scales up on its own with the requests coming in, then drops back to nothing when the traffic dies down.

You need to send structured data to a large language model and get a text response back.
A: LLM API. AWS Bedrock. I just send a request to a model that is already hosted and get the generated text back without training or hosting anything myself.

# Cloud Landscape Q4

A simple data project could be taking data from all recorded soccer games in the world and displaying different stats for the everchanging data. There are thousands of soccer games going on in the world everyday.

My stack would use two different providers from the taxonomy table.

Managed relational DB, Supabase. Soccer data is usually structured by player, score, teams and other metadata so a relational database fits it well, and Supabase gives me a Postgres database without running a server myself.

Serverless compute, AWS Lambda. Lambda runs the backend that queries the database and returns the stats. It does not need a server up 24/7 and it scales on its own when traffic spikes during a big match, then scales back down when it quiets.

There is a benefit to consolidating to one provider. Services on the same platform already talk to each other, so I would deal with one set of credentials, one bill and one dashboard instead of stitching two providers together, and I would spend less time on the glue between them. What I would give up is picking the best tool for each job, since consolidating means taking whatever that one provider offers in every category even when another provider has something cheaper or a better fit. So the convenience costs you some flexibility.
