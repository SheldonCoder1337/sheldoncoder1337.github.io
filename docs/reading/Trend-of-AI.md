---
author: jiale cai
date: 2024-07-28 11:33:05
---

## What is Telemetry?

A nurse in a hospital is far too busy to watch every patient every minute. She relies on telemetry to monitor their vital signs, such as their blood pressure, and alert her if their condition worsens.

Telemetry systems automatically **collect data from sensors**, whether they are attached to a patient, a jet engine or an application server. It then **sends that information to a central site for performance monitoring and to identify problems**.

- 遥测系统是一种自动收集传感器数据，并将该信息发送到中央站点以进行性能监控和识别问题。

Telemetry was developed to automatically measure industrial, scientific and military data **from remote locations**. These included tracking how a missile performed in flight or the temperatures in a blast furnace. In the world of IT and security, telemetry data monitors metrics such as application downtime, database errors, or network connections. This data is the raw material for **observability** – understanding how well applications and services are working, and how users interact with them.

- 遥测数据是可观察性的原材料，用于了解应用程序和服务的运行状况以及用户与它们交互的方式。

## How does Telemetry work?

- When telemetry monitors physical objects, it relies on **sensors** that measure characteristics such as temperature, pressure or vibration.
- When telemetry is used to monitor IT systems, **software agents** gather digital data about performance, uptime and security. They send that data to collectors that process the data and transmit it for storage or analysis.

Telemetry data can be produced in multiple forms by different types of agents. It must thus be **“normalized”** or made to fit a standard structure for use by any analytic tool.

- 因为遥测数据可能有不同的数据来源，所以其必须规范化或者使其适合任何分析工具使用的标准结构。

Historically, normalization was done through a **schema-on-write** process, which required knowing the required format in advance and enforcing that schema before the data was logged. That process is no longer viable given the volume, variety and velocity of data produced by IT infrastructures. A more popular current approach is **schema-on-read**. This converts data into the required format before it is stored and analyzed.

- 过去，规范化操作是通过**写入时模范( schema-on-write)**过程完成的，即事先知道所需的格式，并在记录数据之前强制执行该架构，即传统数据库的模式
- 当下，由于数据规模3V问题，当前更流行的方法是**读取时架构(schema-on-read)**。这会在存储和分析数据之前将数据转换为所需的格式。

## Types of Telemetry

The information produced by IT telemetry data depends on the system being tracked and how the data is used.

- For servers, the data might include how close processors and memory are to being overloaded.
- For networks, it might be latency and bandwidth.For applications and databases, it might be uptime and response time.
- Telemetry designed to detect attacks may include tracking the number of incoming requests to a server, changes to the configuration of an application or a server, or the number or type of files being created or accessed.

Telemetry data comes in three forms:

|Type|Format|Description|Example|
|-|-|-|-|
|Metrics|Numeric values|计量指标，例如处理请求所需的时间、对服务器的传入请求数或失败请求数|`cpu_usage=0.8`|
|Logs|Text|日志文件, 时间序列信息|`2023-04-05T12:34:56Z INFO: User logged in`|
|Traces||显示事务跨基础结构组件（如应用进程、数据库和网络）和服务（如搜索引擎或身份验证机制）所采用的路径||

## How Telemetry is Used

The data gathered by telemetry can provide a real-time view of application performance, so teams can perform root cause analysis on problems, prevent bottlenecks, and identify security threats.

- 遥测收集的数据可以提供应用进程性能的**实时视图(real-time view)**，因此团队可以对问题执行根本原因分析，防止瓶颈并识别安全威胁。

For security monitoring, unusual network traffic patterns might indicate a denial of service attack. Unusual requests for data from an unknown application or repeated unsuccessful attempts to log into a user account may also signal an attempted hack.

Telemetry data can also be used to track how users are interacting with applications and systems. Such user behavior testing can help improve user interfaces and compare whether tweaks to applications and websites can increase user engagement or sales.

- 遥测数据还可用于跟踪用户与应用进程和系统的交互方式。这种用户行为测试可以帮助改进用户界面，并比较对应用进程和网站的调整是否可以提高用户参与度或销售额。

Telemetry data can also help cut costs. By identifying and eliminating underused assets, such as cloud servers that are no longer needed, or helping plan and budget for infrastructure needs by identifying usage trends.

Telemetry from devices on the Internet of Things can do everything from tracking shipments to preventive equipment maintenance. This data can also enable new business models in which a company sells performance, maintenance or production data from equipment in the field.

- 跟踪实时应用进程性能 Track real-time application performance
- 防止安全攻击 Prevent security attacks
- 改善用户体验 Improve user experience
- 跟踪物联网设备的性能和状态 Track the performance and status of IoT devices

## Drawbacks and Challenges of Telemetry

Modern IT infrastructures generate very large data streams in a variety of formats. Not all of this data is critical or even important. It’s easy for system administrators and other IT staff to be overwhelmed by this data, and for storage costs to rise to unacceptable levels.

- 现代 IT 基础设施会生成各种格式的海量数据流。并非所有这些数据都是至关重要的甚至是重要的。系统管理员和其他 IT 员工很容易被这些数据淹没，并且存储成本也会上升到不可接受的水平。

System administrators and software developers must thus decide what data is most important and how to transmit, format and analyze it. Each data transmission method has its pluses and minuses. One option is sending telemetry data directly from the application being monitored. This eliminates the need to run additional software and to manage ports or processes. But if the sending application is complex and generates lots of data, sending that data could bog down the application or network being monitored.

- 因此，系统管理员和软件开发人员必须决定哪些数据最重要，以及如何传输、格式化和分析这些数据。每种数据传输方法都有其优点和缺点。一种选择是直接从被监控的应用进程发送遥测数据。这样就无需运行额外的软件以及管理端口或进程。但是，如果发送应用进程很复杂并且生成大量数据，则发送该数据可能会使正在监视的应用进程或网络陷入困境。

System administrators and software developers must also find ways to minimize the cost of storing telemetry data. One option is to store all the data in a data lake, retrieving only what is needed for analysis when it is needed. Another challenge is how to gather and analyze information from older devices and applications that may not support telemetry. One example is networks that provide performance and health data using the Simple Network Management Protocol.

- 系统管理员和软件开发人员还必须找到最大限度降低遥测数据存储成本的方法。一种选择是将所有数据存储在数据湖中，仅在需要时检索分析所需的数据。另一个挑战是如何收集和分析来自可能不支持遥测的旧设备和应用进程的信息。一个例子是使用简单网络管理协议提供性能和运行状况数据的网络。

Another challenge is finding, acquiring, and deploying analytical tools, including those using artificial intelligence and machine learning, that can sift through Tbytes of data to uncover the incidents and trends that require further attention.

- 另一个挑战是寻找、获取和部署分析工具，包括使用人工智能和机器学习的分析工具，这些工具可以筛选数兆字节的数据，以发现需要进一步关注的事件和趋势。

## Telemetry Tools

Telemetry often relies on software agents running on the source systems to gather the data. In other cases, the source would be an application programming interface (API) to an application or monitoring tool. Connectors then manage the flow of data to multiple destinations and convert it to the protocols and data formats used by various analytical tools. Telemetry data also requires a storage site. This might be a data lake, a time-series database, or a security information and event management (SIEM) system.

- 遥测通常依赖于源系统上运行的软件代理来收集数据。在其他情况下，源可能是应用进程或监视工具的应用进程编程接口 (API)。然后，连接器管理流向多个目的地的数据流，并将其转换为各种分析工具使用的协议和数据格式。遥测数据还需要一个存储站点。这可能是数据湖、时间串行数据库或安全信息和事件管理 (SIEM) 系统。

Given the wide variety of sources of telemetry data, it can be useful to look for tools that comply with **the OpenTelemetry Protocol**, which describes the encoding, transport, and delivery mechanism of telemetry data between telemetry sources and destinations.
