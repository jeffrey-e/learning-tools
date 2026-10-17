// Lines the tests run through the model. DEFAULT_LINE is the page's first preset and the one in python/autoencoder_tf.py.
const DEFAULT_LINE = 'Received disconnect from 85.245.107.41 port 54259:11: disconnected by user';
const LINES = [
  DEFAULT_LINE,
  'Failed password for invalid user admin from 122.189.198.238 port 43749 ssh2',
  "Bad protocol version identification 'GET / HTTP/1.1' from 186.219.213.14 port 59547",
  'Accepted publickey for ubuntu from 85.245.107.41 port 54988 ssh2: RSA SHA256:Kl8kPGZrTiz7g4FO1hyqHdsSBBb5Fge6NWOobN03XJg',
  'User child is on pid 29582',
  // Free text: unknown characters, quotes and backslashes, an emoji, a line longer than 128 characters, one character
  'Accepted password for mallory from 203.0.113.9 port 2222 ssh2 \\ "quoted" é 🙂',
  'x'.repeat(40) + ' Accepted password for elastic_user_0 from 24.151.103.17 port 64635 ssh2 ' + 'y'.repeat(60),
  'A',
];
// A pasted batch: whole auth.log lines (sshd and another program), bare messages, a blank line, a line not in the log.
const BATCH = [
  'Apr 20 14:14:29 ip-10-77-20-248 sshd[3964]: Accepted publickey for ubuntu from 85.245.107.41 port 51816 ssh2: RSA SHA256:Kl8kPGZrTiz7g4FO1hyqHdsSBBb5Fge6NWOobN03XJg',
  'Mar 29 10:32:51 ip-10-77-20-248 CRON[30938]: pam_unix(cron:session): session opened for user root by (uid=0)',
  'Received disconnect from 85.245.107.41 port 54259:11: disconnected by user',
  'Mar 30 01:02:03 ip-10-77-20-248 sshd[123]: Failed password for invalid user admin from 122.189.198.238 port 43749 ssh2',
  '',
  'Invalid user oracle from 203.0.113.50',
  'pam_unix(sshd:session): session opened for user ubuntu by (uid=0)',
  "Bad protocol version identification 'GET / HTTP/1.1' from 186.219.213.14 port 59547",
  'User child is on pid 29582',
  'Mar 31 09:00:00 ip-10-77-20-248 sudo:   ubuntu : TTY=pts/0 ; PWD=/home/ubuntu ; USER=root ; COMMAND=/bin/ls',
].join('\n');
module.exports = { DEFAULT_LINE, LINES, BATCH };
