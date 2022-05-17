red_external = [ 
    '132.197.158.98', # External IP hosting adversarial content
    '202.6.172.98',  # Same as above - external IP for adversary
    '53.192.68.50', # atttacker external DNS
]

red_all = [
    ('132.197.158.98',), # External IP hosting adversarial content
    ('202.6.172.98',),  # Same as above - external IP for adversary
    ('53.192.68.50',), # atttacker external DNS
    ('sysclient0201.systemia.com', '142.20.56.202', '10.20.4.133'),
    ('sysclient0402.systemia.com', '142.20.57.147','10.20.5.88'),
    ('sysclient0660.systemia.com', '142.20.58.149',),
    ('sysclient0104.systemia.com', '142.20.56.105','10.20.4.32'),
    ('sysclient0205.systemia.com', '142.20.56.206','10.20.4.137'),
    ('sysclient0321.systemia.com', '142.20.57.66','10.20.5.3'),
    ('sysclient0255.systemia.com', '142.20.57.0','10.20.4.189'), 
    ('sysclient0355.systemia.com', '142.20.57.100'), 
    ('sysclient0503.systemia.com', '142.20.57.248','10.20.5.193'), 
    ('sysclient0462.systemia.com', '142.20.57.207','10.20.5.150'), 
    ('sysclient0559.systemia.com', '142.20.58.48'), 
    ('sysclient0419.systemia.com', '142.20.57.164','10.20.5.105'), 
    ('sysclient0609.systemia.com', '142.20.58.98','10.20.6.49'),
    ('sysclient0771.systemia.com', '142.20.59.4','10.20.6.217'), 
    ('sysclient0955.systemia.com', '142.20.59.188','10.20.7.155'),
    ('sysclient0874.systemia.com', '142.20.59.107','10.20.7.70'), 
    ('sysclient0170.systemia.com', '142.20.56.171'),
    ('dc1.systemia.com', '142.20.61.130'),
    ('sysclient0203.systemia.com', '142.20.56.204'),
    ('sysclient0974.systemia.com','142.20.59.207'),
    ('sysclient0501.systemia.com', '142.20.57.246','10.20.5.191'),
    ('sysclient0811.systemia.com', '142.20.59.44','10.20.7.5'),
    ('sysclient0010.systemia.com', '142.20.56.11','10.20.3.188'),
    ('sysclient0069.systemia.com', '142.20.56.70'),
    ('sysclient0358.systemia.com', '142.20.57.103'),
    ('sysclient0618.systemia.com', '142.20.58.107','10.20.6.58'),
    ('sysclient0851.systemia.com', '142.20.59.84', '10.20.7.47'), 
    ('sysclient0005.systemia.com',  '142.20.56.6'),
    ('142.20.61.135',), # \\142.20.61.135\share was a file share that was mounted and had data exfil'd out of
    ('sysclient0051.systemia.com','10.20.3.231','142.20.56.52'),
    ('sysclient0351.systemia.com','142.20.57.96','10.20.5.35'),
    ('systemiacom\\zleazer',),
    ('systemiacom\\sysadmin',),
    ('systemiacom\\administrator',), 
    ('systemiacom\\hdorka',),
    ('systemiacom\\bantonio',),  # day 2 initial fishing compromises           
    ('systemiacom\\rsantilli',), # day 2 initial fishing compromises 
    ('systemiacom\\dcoombes',), #compromised user of 0051 on day 3
    ('systemiacom\\bbateman',), # compromised user of 0351 on day 3
]


red_day_3 = [
    ('53.192.68.50',), # atttacker external DNS
    ('sysclient0051.systemia.com','10.20.3.231','142.20.56.52'),
    ('systemiacom\\dcoombes',), # user of 0051
    ('sysclient0351.systemia.com','142.20.57.96','10.20.5.35'),
    ('systemiacom\\bbateman',), # compromised user of 0351 on day 3

    # These users have flow activity to c2 server also
    ('systemiacom\\srivka',),
    ('systemiacom\\csamson',),
    ('systemiacom\\jlopera',),
    ('systemiacom\\plambertz',),
    ('systemiacom\\awalldoff',),
    ('systemiacom\\jlawrence',),
    ('systemiacom\\tcook',),
    ('systemiacom\\rsantilli',),
    ('systemiacom\\bterres',),
]

red_day_3_manual_labels = [
    ('systemiacom\\administrator',), # this user and system below clearly talking to powershell empire at sports.com:443.  powershell downloading files from the url
    ('sysclient0203.systemia.com',),
    ('sysclient0069.systemia.com',),
    ('sysclient0358.systemia.com',),
    ('sysclient0010.systemia.com',),
    ('sysclient0618.systemia.com',),
    ('sysclient0851.systemia.com',),
]


red_day_2 = [
    ('dc1.systemia.com', '142.20.61.130'),
    ('202.6.172.98',),  # Same as above - external IP for adversary
    ('sysclient0501.systemia.com', '142.20.57.246','10.20.5.191'),
    ('sysclient0811.systemia.com', '142.20.59.44','10.20.7.5'),
    ('sysclient0010.systemia.com', '142.20.56.11','10.20.3.188'),
    ('sysclient0069.systemia.com', '142.20.56.70'),
    ('sysclient0203.systemia.com', '142.20.56.204'),
    ('sysclient0358.systemia.com', '142.20.57.103'),
    ('sysclient0618.systemia.com', '142.20.58.107','10.20.6.58'),
    ('sysclient0851.systemia.com', '142.20.59.84', '10.20.7.47'),
    ('systemiacom\\bantonio',),
    ('systemiacom\\rsantilli',),
    ('systemiacom\\sysadmin',),
    ('sysclient0974.systemia.com','142.20.59.207'),
    ('142.20.61.135',), # \\142.20.61.135\share was a file share that was mounted and had data exfil'd out of
    ('sysclient0005.systemia.com',  '142.20.56.6'),  
    ('systemiacom\\administrator',), 
]

red_day_2_manual_labels = [
('142.20.61.136',), # all of these are connections made by bantonio during Deathstar timeperiod
('sysclient0299.systemia.com',),
('sysclient0296.systemia.com',),
('sysclient0278.systemia.com',),
('sysclient0343.systemia.com',),
('sysclient0345.systemia.com',),
('sysclient0347.systemia.com',),
('sysclient0335.systemia.com',),
('sysclient0344.systemia.com',),
('sysclient0450.systemia.com',),
('sysclient0449.systemia.com',),
('sysclient0430.systemia.com',),
('sysclient0428.systemia.com',),
('sysclient0386.systemia.com',),
('sysclient0383.systemia.com',),
('sysclient0385.systemia.com',),
('sysclient0388.systemia.com',),
('sysclient0400.systemia.com',),
('sysclient0399.systemia.com',),
('sysclient0387.systemia.com',),
('sysclient0022.systemia.com',),
('sysclient0020.systemia.com',),
('sysclient0497.systemia.com',),
('sysclient0498.systemia.com',),
('sysclient0499.systemia.com',),
('sysclient0500.systemia.com',),
('sysclient0770.systemia.com',),
('sysclient0075.systemia.com',),
('sysclient0055.systemia.com',),
('sysclient0875.systemia.com',),
('sysclient0406.systemia.com',),
('sysclient0510.systemia.com',),
('sysclient0665.systemia.com',),
('sysclient0268.systemia.com',),
('sysclient0353.systemia.com',),
('sysclient0303.systemia.com',),
('sysclient0201.systemia.com',),
('sysclient0402.systemia.com',),
('sysclient0907.systemia.com',),
('sysclient0868.systemia.com',),
('sysclient0306.systemia.com',),
('sysclient0551.systemia.com',),
('sysclient0307.systemia.com',),
('sysclient0654.systemia.com',),
('sysclient0456.systemia.com',),
('sysclient0451.systemia.com',),
('sysclient0822.systemia.com',),
('sysclient0958.systemia.com',),
('sysclient0870.systemia.com',),
('sysclient0453.systemia.com',),
('sysclient0427.systemia.com',),
('sysclient0405.systemia.com',),
('sysclient0565.systemia.com',),
('sysclient0459.systemia.com',),
('sysclient0964.systemia.com',),
('sysclient0625.systemia.com',),
('sysclient0355.systemia.com',),
('sysclient0524.systemia.com',),
('sysclient0429.systemia.com',),
('sysclient0052.systemia.com',),
('sysclient0561.systemia.com',),
('sysclient0961.systemia.com',),
('sysclient0517.systemia.com',),
('sysclient0871.systemia.com',),
('sysclient0300.systemia.com',),
('sysclient0205.systemia.com',),
('142.20.61.196',),
('142.20.61.191',),
('142.20.61.192',),
('142.20.61.188',),
('142.20.61.190',),
('142.20.61.194',),
('142.20.61.193',),
('142.20.61.195',),
]



red_day_1 = [
    ('132.197.158.98',), # External IP hosting adversarial content
    ('sysclient0201.systemia.com', '142.20.56.202','10.20.4.133'),
    ('sysclient0402.systemia.com', '142.20.57.147','10.20.5.88'),
    ('sysclient0660.systemia.com', '142.20.58.149',),
    ('sysclient0104.systemia.com', '142.20.56.105','10.20.4.32'),
    ('sysclient0205.systemia.com', '142.20.56.206','10.20.4.137'),
    ('sysclient0321.systemia.com', '142.20.57.66','10.20.5.3'),
    ('sysclient0255.systemia.com', '142.20.57.0','10.20.4.189'),
    ('sysclient0355.systemia.com', '142.20.57.100'),
    ('sysclient0503.systemia.com', '142.20.57.248','10.20.5.193'),
    ('sysclient0462.systemia.com', '142.20.57.207','10.20.5.150'),
    ('sysclient0559.systemia.com', '142.20.58.48'),
    ('sysclient0419.systemia.com', '142.20.57.164','10.20.5.105'),
    ('sysclient0609.systemia.com', '142.20.58.98','10.20.6.49'),
    ('sysclient0771.systemia.com', '142.20.59.4','10.20.6.217'),
    ('sysclient0955.systemia.com', '142.20.59.188','10.20.7.155'),
    ('sysclient0874.systemia.com', '142.20.59.107','10.20.7.70'),      
    ('sysclient0170.systemia.com', '142.20.56.171'),
    ('sysclient0203.systemia.com', '142.20.56.204'), # SMB attempt also some ndetwork activity from zlezer
    ('dc1.systemia.com', '142.20.61.130'),
    ('systemiacom\\hdorka',),
    ('systemiacom\\zleazer',),
]


red_day_1_manual_labels = [
    ('systemiacom\\administrator',), # very obvious administrator use compromise on 0660 and 0402 as seen above
]




red_team_day_1 = [
    {"host_name": "sysclient0201.systemia.com", "src_ip":"142.20.56.202", "pid": 2952, "start_timestamp": "2019-09-23T11:26:38.000-04:00"},
    {"host_name": "sysclient0201.systemia.com", "src_ip":"142.20.56.202", "pid": 5452, "start_timestamp": "2019-09-23T11:23:29.000-04:00"},
    {"host_name": "sysclient0402.systemia.com", "src_ip":"142.20.57.147", "pid": 3168, "start_timestamp": "2019-09-23T13:25:41.000-04:00"},
    {"host_name": "sysclient0660.systemia.com", "src_ip":"142.20.58.149", "pid": 880, "start_timestamp": "2019-09-23T13:38:31.000-04:00"},
    {"host_name": "sysclient0104.systemia.com", "src_ip":"142.20.56.105", "pid": 3160, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0205.systemia.com", "src_ip":"142.20.56.206", "pid": 5012, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0321.systemia.com", "src_ip":"142.20.57.66", "pid": 2980, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0255.systemia.com", "src_ip":"142.20.57.0", "pid": 3472, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0355.systemia.com", "src_ip":"142.20.57.100", "pid": 1884, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0503.systemia.com", "src_ip":"142.20.57.248", "pid": 1472, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0462.systemia.com", "src_ip":"142.20.57.207", "pid": 2536, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0559.systemia.com", "src_ip":"142.20.58.48", "pid": 1400, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0419.systemia.com", "src_ip":"142.20.57.164", "pid": 1700, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0609.systemia.com", "src_ip":"142.20.58.98", "pid": 3460, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0771.systemia.com", "src_ip":"142.20.59.4", "pid": 4244, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0955.systemia.com", "src_ip":"142.20.59.188", "pid": 4760, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0874.systemia.com", "src_ip":"142.20.59.107", "pid": 5224, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "sysclient0170.systemia.com", "src_ip":"142.20.56.171", "pid": 644, "start_timestamp": "2019-09-23T14:45:13.000-04:00"},
    {"host_name": "dc1.systemia.com", "src_ip":"142.20.61.130", "pid": 1852, "start_timestamp": "2019-09-23T14:04:45.000-04:00"}]

anom_ips = ['142.20.56.202', '142.20.57.147', '142.20.58.149', '142.20.56.105', '142.20.56.206', '142.20.57.66', '142.20.57.0', '142.20.57.100', '142.20.57.248', '142.20.57.207', '142.20.58.48', '142.20.57.164', '142.20.58.98', '142.20.59.4', '142.20.59.188', '142.20.59.107', '142.20.56.171',  '142.20.61.130', '142.20.61.130', '142.20.57.246', '142.20.59.44', '142.20.56.11', '142.20.56.70', '142.20.56.204', '142.20.57.103', '142.20.58.107', '142.20.59.84']

red_team_day_2 = [
    {"host_name": "sysclient0501.systemia.com", "src_ip":"142.20.57.246", "pid": 648, "start_timestamp": "2019-09-24T10:46:02.000-04:00"},
    {"host_name": "sysclient0501.systemia.com", "src_ip":"142.20.57.246", "pid": 5076,"start_timestamp": "2019-09-24T11:26:34.000-04:00"},
    {"host_name": "sysclient0501.systemia.com", "src_ip":"142.20.57.246", "pid": 1748, "start_timestamp": "2019-09-24T11:34:56.000-04:00"},
    {"host_name": "sysclient0811.systemia.com", "src_ip":"142.20.59.44", "pid": 3780, "start_timestamp": "2019-09-24T14:45:13.000-04:00"},
    {"host_name": "dc1.systemia.com", "src_ip":"142.20.61.130", "pid": 3880, "start_timestamp": "2019-09-24T11:31:17.000-04:00"},
    {"host_name": "sysclient0010.systemia.com", "src_ip":"142.20.56.11", "pid": 3584, "start_timestamp": "2019-09-24T15:42:36.000-04:00"},
    {"host_name": "sysclient0069.systemia.com", "src_ip":"142.20.56.70", "pid": 4152, "start_timestamp": "2019-09-24T15:42:36.000-04:00"},
    {"host_name": "sysclient0203.systemia.com", "src_ip":"142.20.56.204", "pid": 5388, "start_timestamp": "2019-09-24T15:42:36.000-04:00"},
    {"host_name": "sysclient0358.systemia.com", "src_ip":"142.20.57.103", "pid": 2984, "start_timestamp": "2019-09-24T15:42:36.000-04:00"},
    {"host_name": "sysclient0618.systemia.com", "src_ip":"142.20.58.107", "pid": 4060, "start_timestamp": "2019-09-24T15:42:36.000-04:00"},
    {"host_name": "sysclient0851.systemia.com", "src_ip":"142.20.59.84", "pid": 4652, "start_timestamp": "2019-09-24T15:42:36.000-04:00"}]


red_team_day_3 = [
    {"host_name": "sysclient0051.systemia.com", "src_ip":"142.20.56.52", "pid": 2712, "start_timestamp": "2019-09-25T10:29:42.000-04:00"},
    {"host_name": "sysclient0051.systemia.com", "src_ip":"142.20.56.52", "pid": 568,"start_timestamp": "2019-09-24T10:40:49.000-04:00"},
    {"host_name": "sysclient0351.systemia.com", "src_ip":"142.20.57.96", "pid": 1932, "start_timestamp": "2019-09-24T11:23:31.000-04:00"},
    {"host_name": "sysclient0351.systemia.com", "src_ip":"142.20.57.96", "pid": 1256, "start_timestamp": "2019-09-24T11:24:30.000-04:00"}]


anom_host_day_1 = ["sysclient0201.systemia.com",  "sysclient0402.systemia.com", "sysclient0660.systemia.com",  "sysclient0104.systemia.com", "sysclient0205.systemia.com", "sysclient0321.systemia.com", "sysclient0255.systemia.com", "sysclient0355.systemia.com", "sysclient0503.systemia.com", "sysclient0462.systemia.com", "sysclient0559.systemia.com",  "sysclient0419.systemia.com", "sysclient0609.systemia.com", "sysclient0771.systemia.com", "sysclient0955.systemia.com",  "sysclient0874.systemia.com", "sysclient0170.systemia.com", "dc1.systemia.com"]

anom_host_day_2 = ["sysclient0501.systemia.com", "sysclient0811.systemia.com",  "dc1.systemia.com", "sysclient0010.systemia.com","sysclient0069.systemia.com", "sysclient0203.systemia.com", "sysclient0358.systemia.com", "sysclient0618.systemia.com", "sysclient0851.systemia.com"]

anom_host_day_3 = ["sysclient0051.systemia.com", "sysclient0351.systemia.com"]

