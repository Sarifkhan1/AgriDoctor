# 🌿 AgriDoctor AI - VPS Multi-Site Deployment & SSL Guide

This guide provides a professional, DevOps-grade walkthrough for deploying **AgriDoctor AI** on your VPS server (**69.62.78.148**) alongside **existing hosted websites** without causing port binding conflicts or service disruption, under the domain **`agridoctor.cloud`**.

---

## 🏗️ Multi-Site DevOps Architecture

On a shared VPS hosting multiple sites, port `80` (HTTP) and port `443` (HTTPS) are already bound by the VPS's **Primary Reverse Proxy (Edge Proxy)**. 

To deploy AgriDoctor without conflicts, we map its Docker services to alternative ports internally, and configure the host's primary Nginx to terminate SSL and route `agridoctor.cloud` traffic:

```mermaid
graph TD
    User([User Browser]) -->|HTTPS Port 443| HostProxy[VPS Host Nginx Edge Proxy]
    User -->|HTTP Port 80| HostProxy
    
    subgraph Docker Internal Network
        HostProxy -->|Proxy for agridoctor.cloud| ProjectProxy[AgriDoctor Nginx:8020]
        ProjectProxy -->|Serve Static Files| WebRoot[/usr/share/nginx/html]
        ProjectProxy -->|Proxy /api/*| FastAPI[FastAPI Backend:8002]
        FastAPI -->|Write/Read| SQLite[(SQLite Database)]
    end
```

---

## 🔒 Port Allocation Summary

To avoid conflict with other apps (like phpMyAdmin or other websites), AgriDoctor uses these conflict-free host ports:

| Service | Internal Port | Host Port (Conflict-Free) | Accessibility |
| :--- | :--- | :--- | :--- |
| **Frontend (Nginx)** | `80` | **`8020`** | Proxied by VPS edge proxy |
| **Backend (FastAPI)** | `8000` | **`8002`** | Private / internal docker network |
| **Annotator (Streamlit)**| `8501` | **`8502`** | Internal data labeling tool |

---

## ⚡ Single-Click Deployment (Recommended)

Run the automated deployer from your **local Mac terminal** to sync the project and spin up the containers on their safe, alternative ports:

```bash
cd /Users/santosh/Documents/agri-doctor
./deploy-vps.sh
```
*(Enter your root password when prompted: `64X1i7jkQg1@RO--y)L7`)*

---

## 🛠️ VPS Edge Proxy Setup & SSL Provisioning

Once the containers are running on port `8020`, configure your VPS's primary reverse proxy and secure it with Let's Encrypt SSL.

### Case A: VPS Runs Nginx on the Host (Standard)

Connect to your VPS via SSH on port `2222`:
```bash
ssh -p 2222 root@69.62.78.148
```

#### Step 1: Copy host Nginx server block
Copy our custom `agridoctor.cloud.conf` to Nginx's sites-available:
```bash
cp /root/agri-doctor/agridoctor.cloud.conf /etc/nginx/sites-available/agridoctor.cloud
```

#### Step 2: Enable the site
Link the file to sites-enabled:
```bash
ln -sf /etc/nginx/sites-available/agridoctor.cloud /etc/nginx/sites-enabled/
```

#### Step 3: Test and reload Nginx
Verify the config has no syntax errors and reload Nginx:
```bash
nginx -t
nginx -s reload
```

#### Step 4: Obtain Let's Encrypt SSL on Host
Run Certbot directly on the host to generate the SSL certificate and automatically update the Nginx configuration to enable HTTPS:
```bash
certbot --nginx -d agridoctor.cloud -d www.agridoctor.cloud
```
*(Select the option to automatically redirect all HTTP traffic to HTTPS if prompted)*

---

### Case B: VPS Runs Nginx Proxy Manager (NPM UI)

If your VPS uses Nginx Proxy Manager to manage websites visually:

1. Log in to your NPM Admin Panel (typically on port `81` or similar).
2. Go to **Hosts** > **Proxy Hosts** > **Add Proxy Host**.
3. Fill in the form:
   - **Domain Names**: `agridoctor.cloud`, `www.agridoctor.cloud`
   - **Scheme**: `http`
   - **Forward Name/IP**: `127.0.0.1` (or your VPS's internal docker bridge gateway IP, e.g., `172.17.0.1`)
   - **Forward Port**: **`8020`**
   - Enable **Block Common Exploits** and **Websockets Support**.
4. Go to the **SSL** tab:
   - Select **Request a new SSL Certificate** from Let's Encrypt.
   - Check **Force SSL** and **HTTP/2 Support**.
   - Accept the Terms of Service.
5. Click **Save**. NPM will handle certificate generation and HTTPS routing instantly!
