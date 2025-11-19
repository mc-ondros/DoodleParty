class ModerationShield {
    constructor() {
        this.moderationMessage = '';
        this.moderationElement = document.createElement('div');
        this.moderationElement.className = 'moderation-shield';
        document.body.appendChild(this.moderationElement);
    }

    displayMessage(message) {
        this.moderationMessage = message;
        this.updateUI();
    }

    clearMessage() {
        this.moderationMessage = '';
        this.updateUI();
    }

    updateUI() {
        this.moderationElement.textContent = this.moderationMessage;
        this.moderationElement.style.display = this.moderationMessage ? 'block' : 'none';
    }
}

export default ModerationShield;