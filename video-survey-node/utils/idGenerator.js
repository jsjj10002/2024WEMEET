class IdGenerator {
    constructor() {
        this.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    }

    generateNextId(lastId) {
        if (!lastId) return 'AA000';

        let letters = lastId.substring(0, 2);
        let numbers = parseInt(lastId.substring(2));

        if (numbers < 999) {
            // 숫자만 증가
            return letters + String(numbers + 1).padStart(3, '0');
        }

        // 숫자가 999이면 문자를 변경
        numbers = 0;
        let firstLetter = letters[0];
        let secondLetter = letters[1];

        if (secondLetter === 'Z') {
            firstLetter = this.letters[this.letters.indexOf(firstLetter) + 1];
            secondLetter = 'A';
        } else {
            secondLetter = this.letters[this.letters.indexOf(secondLetter) + 1];
        }

        return `${firstLetter}${secondLetter}${String(numbers).padStart(3, '0')}`;
    }
}

module.exports = new IdGenerator();