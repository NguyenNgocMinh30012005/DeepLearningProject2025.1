# 🔑 SSH Setup Guide for GitHub

## Remote URL Changed
✅ Changed from HTTPS to SSH:
- Old: `https://github.com/NguyenNgocMinh30012005/DeepLearningProject2025.1.git`
- New: `git@github.com:NguyenNgocMinh30012005/DeepLearningProject2025.1.git`

---

## Step-by-Step SSH Setup

### **Step 1: Generate SSH Key**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

When prompted:
- **File location:** Press `Enter` (use default: `~/.ssh/id_ed25519`)
- **Passphrase:** Press `Enter` (no passphrase) or enter one for extra security
- **Confirm passphrase:** Press `Enter` again

Output should show:
```
Your identification has been saved in /root/.ssh/id_ed25519
Your public key has been saved in /root/.ssh/id_ed25519.pub
```

---

### **Step 2: Copy Your Public Key**

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the ENTIRE output (starts with `ssh-ed25519 AAAA...`)

Example output:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJl3dIeudNqd0DPMRD6OIh65A9pu9hj/example your_email@example.com
```

---

### **Step 3: Add SSH Key to GitHub**

1. Go to: **https://github.com/settings/keys**
2. Click **"New SSH key"** button
3. Fill in:
   - **Title:** `My Workspace` (or any name)
   - **Key type:** `Authentication Key`
   - **Key:** Paste the public key from Step 2
4. Click **"Add SSH key"**
5. Enter your GitHub password if prompted

---

### **Step 4: Test SSH Connection**

```bash
ssh -T git@github.com
```

**First time:** You'll see:
```
The authenticity of host 'github.com' can't be established.
Are you sure you want to continue connecting (yes/no)?
```
Type: `yes` and press Enter

**Expected success message:**
```
Hi NguyenNgocMinh30012005! You've successfully authenticated, but GitHub does not provide shell access.
```

If you see this, SSH is working! ✅

---

### **Step 5: Push to GitHub**

```bash
cd /workspace
git push -u origin main
```

Should push successfully without asking for password! 🎉

---

## Troubleshooting

### **Error: Permission denied (publickey)**
- Make sure you copied the ENTIRE public key
- Verify key was added to: https://github.com/settings/keys
- Try: `ssh -T git@github.com` to test

### **Error: Could not resolve hostname**
- Check internet connection
- Verify URL: `git remote -v`

### **Want to use HTTPS instead?**
```bash
git remote set-url origin https://github.com/NguyenNgocMinh30012005/DeepLearningProject2025.1.git
```
Then use Personal Access Token for authentication.

---

## Quick Reference

```bash
# Generate key
ssh-keygen -t ed25519 -C "email@example.com"

# View public key
cat ~/.ssh/id_ed25519.pub

# Test connection
ssh -T git@github.com

# Push code
git push -u origin main
```

---

**Need help?** Check GitHub's SSH guide: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
