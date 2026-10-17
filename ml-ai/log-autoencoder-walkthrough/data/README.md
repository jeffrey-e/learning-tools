# Data

`auth.log` is copied unmodified from Elastic's examples repository:

https://github.com/elastic/examples/blob/master/Machine%20Learning/Security%20Analytics%20Recipes/suspicious_login_activity/data/auth.log

It's part of the "Detect suspicious login activity" security analytics recipe. Copyright Elasticsearch B.V., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

The file is the authentication log of one internet-facing Ubuntu server, 27 March to 20 April: 7,121 lines, 4,095 of them from sshd. It has no labels; the walkthrough labels sshd messages by rule (see `RULES` in `../model.js`).
