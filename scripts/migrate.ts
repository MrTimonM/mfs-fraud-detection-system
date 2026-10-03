import {migrate,database} from '../src/lib/store';
await migrate();await database().end();console.log('Database schema and default rules are ready.');
