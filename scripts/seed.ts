import {withState,database} from '../src/lib/store';
import {seedState} from '../src/lib/scenarios';
await withState(true,s=>{if(s.transactions.length)throw new Error('Seed refused: database already has transactions.');Object.assign(s,seedState());});
await database().end();console.log('Inserted 48 synthetic demo transactions.');
