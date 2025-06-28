// Worker.js
// import MavlinkParser from 'mavlinkParser'
const mavparser = require('./mavlinkParser')
const DataflashParser = require('./JsDataflashParser/parser').default
const DjiParser = require('./djiParser').default

let parser
self.addEventListener('message', async function (event) {
    if (event.data === null) {
        console.log('got bad file message!')
    } else if (event.data.action === 'parse') {
        const data = event.data.file
        if (event.data.isTlog) {
            parser = new mavparser.MavlinkParser()
            parser.processData(data)
        } else if (event.data.isDji) {
            parser = new DjiParser()
            await parser.processData(data)
        } else {
            parser = new DataflashParser(true)
            // First discover all available message types
            parser.processData(data)
            
            // Get all available message types and parse them
            const allMessageTypes = Object.keys(parser.messageTypes || {})
            console.log('All discovered message types:', allMessageTypes)
            
            // Parse all available message types
            for (const msgType of allMessageTypes) {
                try {
                    parser.parseAtOffset(msgType)
                } catch (error) {
                    console.warn(`Failed to parse message type ${msgType}:`, error)
                }
            }
            
            // Process files if FILE messages were parsed
            if (parser.messages && parser.messages.FILE) {
                parser.processFiles()
            }
        }

    } else if (event.data.action === 'loadType') {
        if (!parser) {
            console.log('parser not ready')
            return
        }
        parser.loadType(event.data.type.split('[')[0])
    } else if (event.data.action === 'trimFile') {
        parser.trimFile(event.data.time)
    }
})
