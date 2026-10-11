# Part 27 - AIT window length and sample mode

Statistics only. No training, no ENIV, no accuracy: every number below is a count or a fraction of counts, produced by `experiments/ait_window/run_window_table.py` over the Part 26 loader.

`attack_fraction` is over **label_known samples only** - a window no label-covered file contributed to is unexamined, not benign.

## 1. Samples and labels

| scenario | mode | window s | grid | samples | attack | benign | not labelled | attack frac (labelled) | dnsteal-only attack | dnsteal-only frac | attack w/ other step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| russellmitchell | network | 30 | 11,520 | 11,520 | 10,305 | 473 | 742 | 0.956 | 10,292 | 0.999 | 13 |
| russellmitchell | host | 30 | 11,520 | 92,032 | 10,315 | 1,066 | 80,651 | 0.906 | 10,298 | 0.998 | 17 |
| russellmitchell | network | 60 | 5,760 | 5,760 | 5,153 | 359 | 248 | 0.935 | 5,143 | 0.998 | 10 |
| russellmitchell | host | 60 | 5,760 | 46,431 | 5,160 | 930 | 40,341 | 0.847 | 5,147 | 0.997 | 13 |
| russellmitchell | network | 300 | 1,152 | 1,152 | 1,031 | 114 | 7 | 0.900 | 1,026 | 0.995 | 5 |
| russellmitchell | host | 300 | 1,152 | 9,834 | 1,034 | 613 | 8,187 | 0.628 | 1,028 | 0.994 | 6 |
| russellmitchell | network | 900 | 384 | 384 | 344 | 40 | 0 | 0.896 | 339 | 0.985 | 5 |
| russellmitchell | host | 900 | 384 | 3,663 | 347 | 461 | 2,855 | 0.429 | 341 | 0.983 | 6 |
| santos | network | 30 | 11,520 | 11,520 | 6,645 | 2,328 | 2,547 | 0.741 | 6,632 | 0.998 | 13 |
| santos | host | 30 | 11,520 | 90,323 | 6,649 | 2,955 | 80,719 | 0.692 | 6,633 | 0.998 | 16 |
| santos | network | 60 | 5,760 | 5,760 | 3,324 | 1,595 | 841 | 0.676 | 3,316 | 0.998 | 8 |
| santos | host | 60 | 5,760 | 45,840 | 3,328 | 2,223 | 40,289 | 0.600 | 3,317 | 0.997 | 11 |
| santos | network | 300 | 1,152 | 1,152 | 667 | 441 | 44 | 0.602 | 663 | 0.994 | 4 |
| santos | host | 300 | 1,152 | 9,779 | 669 | 939 | 8,171 | 0.416 | 664 | 0.993 | 5 |
| santos | network | 900 | 384 | 384 | 224 | 160 | 0 | 0.583 | 221 | 0.987 | 3 |
| santos | host | 900 | 384 | 3,658 | 226 | 581 | 2,851 | 0.280 | 222 | 0.982 | 4 |

dnsteal-family = `attacker`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`. An attack window counted in `dnsteal-only` carries no other attack step.

## 2. Contiguous runs on the time grid

The run **count** is how many independent attack episodes the row contains; the lengths are how long each lasted, in windows. Host mode runs are per host and pooled. A dropped empty window breaks a run.

| scenario | mode | window s | attack runs | attack run lengths | benign runs | benign run lengths |
|---|---|---|---|---|---|---|
| russellmitchell | network | 30 | **1** | 10305 | 253 | 12/7/6/6/6/5/5/4/... (+245 more) |
| russellmitchell | host | 30 | **4** | 10305/7/2/1 | 805 | 12/7/6/6/5/5/5/5/... (+797 more) |
| russellmitchell | network | 60 | **1** | 5153 | 132 | 18/15/14/11/10/10/8/8/... (+124 more) |
| russellmitchell | host | 60 | **4** | 5153/4/2/1 | 655 | 18/15/14/11/10/8/7/7/... (+647 more) |
| russellmitchell | network | 300 | **1** | 1031 | 8 | 99/5/3/3/1/1/1/1 |
| russellmitchell | host | 300 | **4** | 1031/1/1/1 | 422 | 99/13/11/7/6/5/5/5/... (+414 more) |
| russellmitchell | network | 900 | **1** | 344 | 1 | 40 |
| russellmitchell | host | 900 | **4** | 344/1/1/1 | 177 | 40/39/11/11/7/7/7/7/... (+169 more) |
| santos | network | 30 | **5** | 6633/8/2/1/1 | 856 | 35/25/21/21/19/18/18/17/... (+848 more) |
| santos | host | 30 | **8** | 6633/7/3/2/1/1/1/1 | 1494 | 34/21/18/18/18/18/16/14/... (+1486 more) |
| santos | network | 60 | **4** | 3317/4/2/1 | 324 | 74/73/68/64/59/56/45/43/... (+316 more) |
| santos | host | 60 | **7** | 3317/4/2/2/1/1/1 | 856 | 73/68/64/56/56/38/37/35/... (+848 more) |
| santos | network | 300 | **3** | 664/2/1 | 37 | 168/98/76/18/7/7/6/6/... (+29 more) |
| santos | host | 300 | **5** | 664/2/1/1/1 | 437 | 168/105/76/28/11/10/10/9/... (+429 more) |
| santos | network | 900 | **3** | 222/1/1 | 3 | 111/48/1 |
| santos | host | 900 | **5** | 222/1/1/1/1 | 183 | 66/50/28/27/13/11/7/7/... (+175 more) |

## 3. Per-view missing fraction (view_mask False)

| scenario | mode | window s | mean views present | apache | auth | dns | audit | suricata |
|---|---|---|---|---|---|---|---|---|
| russellmitchell | network | 30 | 2.95 | 0.248 | 0.936 | 0.057 | 0.809 | 0.000 |
| russellmitchell | host | 30 | 1.36 | 0.815 | 0.971 | 0.766 | 0.962 | 0.124 |
| russellmitchell | network | 60 | 3.17 | 0.225 | 0.874 | 0.036 | 0.694 | 0.000 |
| russellmitchell | host | 60 | 1.42 | 0.808 | 0.943 | 0.764 | 0.933 | 0.132 |
| russellmitchell | network | 300 | 4.05 | 0.185 | 0.477 | 0.003 | 0.284 | 0.000 |
| russellmitchell | host | 300 | 1.75 | 0.791 | 0.742 | 0.767 | 0.775 | 0.180 |
| russellmitchell | network | 900 | 4.84 | 0.161 | 0.000 | 0.000 | 0.000 | 0.000 |
| russellmitchell | host | 900 | 2.22 | 0.791 | 0.382 | 0.790 | 0.547 | 0.266 |
| santos | network | 30 | 2.81 | 0.368 | 0.936 | 0.185 | 0.697 | 0.000 |
| santos | host | 30 | 1.33 | 0.847 | 0.970 | 0.803 | 0.944 | 0.107 |
| santos | network | 60 | 3.09 | 0.337 | 0.873 | 0.124 | 0.572 | 0.000 |
| santos | host | 60 | 1.40 | 0.835 | 0.942 | 0.788 | 0.915 | 0.120 |
| santos | network | 300 | 3.99 | 0.298 | 0.477 | 0.031 | 0.204 | 0.000 |
| santos | host | 300 | 1.74 | 0.815 | 0.740 | 0.778 | 0.757 | 0.175 |
| santos | network | 900 | 4.71 | 0.286 | 0.000 | 0.000 | 0.000 | 0.000 |
| santos | host | 900 | 2.22 | 0.810 | 0.382 | 0.792 | 0.535 | 0.265 |

## 4. Per-view mean events per window where the view is present

In-grid events only, summed from the scan's cells, so events outside the declared scenario interval cannot inflate a per-window mean.

| scenario | mode | window s | apache | auth | dns | audit | suricata |
|---|---|---|---|---|---|---|---|
| russellmitchell | network | 30 | 11.9 | 7.2 | 46.3 | 8.5 | 85.5 |
| russellmitchell | host | 30 | 6.1 | 2.0 | 23.4 | 5.3 | 12.2 |
| russellmitchell | network | 60 | 23.2 | 7.4 | 90.7 | 10.6 | 171.1 |
| russellmitchell | host | 60 | 11.6 | 2.0 | 45.9 | 6.0 | 24.4 |
| russellmitchell | network | 300 | 110.1 | 8.8 | 438.2 | 22.7 | 855.5 |
| russellmitchell | host | 300 | 50.3 | 2.1 | 219.5 | 8.5 | 122.2 |
| russellmitchell | network | 900 | 320.9 | 13.9 | 1,311.2 | 48.8 | 2,566.5 |
| russellmitchell | host | 900 | 135.1 | 2.4 | 655.6 | 11.3 | 366.6 |
| santos | network | 30 | 28.9 | 7.2 | 56.6 | 6.9 | 87.5 |
| santos | host | 30 | 15.2 | 2.0 | 29.8 | 4.8 | 12.5 |
| santos | network | 60 | 55.0 | 7.2 | 105.3 | 9.7 | 174.9 |
| santos | host | 60 | 27.8 | 2.0 | 54.8 | 6.2 | 25.0 |
| santos | network | 300 | 259.7 | 8.8 | 476.2 | 26.2 | 874.6 |
| santos | host | 300 | 115.9 | 2.1 | 244.4 | 10.1 | 124.9 |
| santos | network | 900 | 766.9 | 13.8 | 1,384.0 | 62.6 | 2,623.9 |
| santos | host | 900 | 301.9 | 2.4 | 698.4 | 14.1 | 374.8 |

## 5. Attack steps reaching at least one attack window

| scenario | mode | window s | distinct steps | steps |
|---|---|---|---|---|
| russellmitchell | network | 30 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | host | 30 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | network | 60 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | host | 60 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | network | 300 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | host | 300 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | network | 900 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| russellmitchell | host | 900 | 20 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-dropped`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | network | 30 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | host | 30 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | network | 60 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | host | 60 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | network | 300 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | host | 300 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | network | 900 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |
| santos | host | 900 | 19 | `attacker`, `attacker_change_user`, `attacker_http`, `dirb`, `dns_scan`, `dnsteal`, `dnsteal-received`, `escalate`, `escalated_command`, `escalated_sudo_command`, `escalated_sudo_session`, `exfiltration-service`, `foothold`, `network_scan`, `service_scan`, `traceroute`, `webshell_cmd`, `webshell_upload`, `wpscan` |

## 6. Are the views identical across the two scenarios?

**Yes.** Compared as (source name, native feature width) in view order, for every row.

- `russellmitchell`: `apache`:9, `auth`:8, `dns`:9, `audit`:9, `suricata`:10
- `santos`: `apache`:9, `auth`:8, `dns`:9, `audit`:9, `suricata`:10

Same-sensor groups: `{'network': [4]}` (view indices).

## 7. Smallest window leaving at least 30 of each class

`label_known` samples only. A reported threshold, not a choice: the window and mode decision stays with the user at the review gate.

| scenario | mode | smallest usable window s | attack there | benign there | windows tried |
|---|---|---|---|---|---|
| russellmitchell | host | **30** | 10,315 | 1,066 | 30, 60, 300, 900 |
| russellmitchell | network | **30** | 10,305 | 473 | 30, 60, 300, 900 |
| santos | host | **30** | 6,649 | 2,955 | 30, 60, 300, 900 |
| santos | network | **30** | 6,645 | 2,328 | 30, 60, 300, 900 |

## 8. Provenance

| scenario | archive | archive size | entries fetched | fetch complete |
|---|---|---|---|---|
| russellmitchell | `russellmitchell.zip` | 7,132.7 MB | 91 | True |
| santos | `santos.zip` | 10,004.8 MB | 86 | True |

Raw and processed data are gitignored; the CRC-32 and SHA-256 in each `data/raw/ait/<scenario>/manifest.json` tie these numbers to specific bytes.

