const mongoose = require('mongoose');

const SurveySchema = new mongoose.Schema({
    customerId: {
        type: String,
        required: true
    },
    taste: {
        type: Number,
        required: true,
        min: 1,
        max: 5
    },
    seasoning: {
        type: String,
        required: true,
        enum: ['매우 싱거움', '싱거움', '적당함', '짬', '매우 짬']
    },
    service: {
        type: Number,
        required: true,
        min: 1,
        max: 5
    },
    cleanliness: {
        type: Number,
        required: true,
        min: 1,
        max: 5
    },
    portion: {
        type: String,
        required: true,
        enum: ['매우 적음', '적음', '적당함', '많음', '매우 많음']
    },
    feedback: {
        type: String
    }
});

module.exports = mongoose.model('Survey', SurveySchema);